import os
import sys
import shutil
import math
import time

# 优化 CUDA 显存管理器，消除碎片化 OOM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch

# 限制显存占用上限，防止占满整卡
torch.cuda.set_per_process_memory_fraction(0.65)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from torch.utils.data import IterableDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import load_from_disk
from dataclasses import dataclass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from llm.llm_model import NaviLLM, ModelArgs
from dataset.navi_token import NaviTokenizer

# ================= 路径配置 =================
LOGS_DIR = os.path.join(BASE_DIR, "logs", "navi_llm")
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "skypile_100b")

CHKPT_PATH = os.path.join(MODEL_DIR, "checkpoint.pth")
BEST_PATH = os.path.join(MODEL_DIR, "best_model.pth")
FINAL_PATH = os.path.join(MODEL_DIR, "final_model.pth")

# ================= 训练超参数 =================
@dataclass
class TrainArgs:
    micro_batch_size: int = 2            # 2， 减少 kernel launch 频率
    gradient_accumulation_steps: int = 16 # 16, 等效 batch 保持 32
    total_steps: int = 300000                # ~18B tokens (19 tokens/param, Chinchilla)
    cosine_steps: int = 200000               # 余弦退火终点 (71% total)
    warmup_steps: int = 4000                 # warmup 占 cosine 的 2%

    learning_rate: float = 3e-4              
    min_lr: float = 3e-6                     
    final_lr: float = 1e-6                   

    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    save_interval: int = 5000               
    eval_interval: int = 20000                
    eval_iters: int = 100                    

# ================= 数据集 =================
class PackedIterableDataset(IterableDataset):
    def __init__(self, hf_dataset, tokenizer, seq_len):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self):
        buffer = []
        for item in self.dataset:
            text = item.get("text", "").strip()
            if not text: continue
            tokens = self.tokenizer.encode(text, add_bos=True, add_eos=True)
            buffer.extend(tokens)

            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len + 1:]
                yield (
                    torch.tensor(chunk[:-1], dtype=torch.long),
                    torch.tensor(chunk[1:], dtype=torch.long)
                )

# ================= 核心工具函数 =================
def setup_directories(resume=False):
    if not resume and os.path.exists(LOGS_DIR):
        print(f"[Setup] 清理旧日志，执行干净初始化: {LOGS_DIR}")
        shutil.rmtree(LOGS_DIR)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

def get_lr(global_step, args: TrainArgs):
    if global_step < args.warmup_steps:
        return args.learning_rate * (global_step + 1) / args.warmup_steps
    if global_step >= args.cosine_steps:
        return args.final_lr
    decay_ratio = (global_step - args.warmup_steps) / (args.cosine_steps - args.warmup_steps)
    return args.min_lr + 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) * (args.learning_rate - args.min_lr)

def calculate_accuracy(logits, targets, ignore_index=-100):
    preds = torch.argmax(logits, dim=-1)
    valid_mask = (targets != ignore_index)
    correct = (preds == targets) & valid_mask
    acc = correct.sum().float() / valid_mask.sum().float()
    return acc.item()

def log_metrics(writer, step, loss, acc, lr, grad_norm, throughput):
    writer.add_scalar("Train/Loss", loss, step)
    writer.add_scalar("Train/Accuracy", acc, step)
    writer.add_scalar("Train/LR", lr, step)
    writer.add_scalar("Perf/Tokens_per_Sec", throughput, step)
    if grad_norm is not None:
        writer.add_scalar("Train/GradNorm", grad_norm, step)

def log_detailed_metrics(writer, model, tokenizer, step, train_args):
    # 提取底层原始模型 (剥离 torch.compile 的包裹壳)
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    
    for name, param in raw_model.named_parameters():
        if "layers.0" in name or f"layers.{raw_model.args.n_layers - 1}" in name or "tok_embeddings" in name:
            writer.add_histogram(f"Weights/{name}", param.detach().cpu(), step)
            if param.grad is not None:
                writer.add_histogram(f"Gradients/{name}", param.grad.detach().cpu(), step)

    total_norm = sum(p.norm(2).item() ** 2 for p in raw_model.parameters() if p is not None)
    writer.add_scalar("Debug/ParamNorm", total_norm ** 0.5, step)

    proj_size = 2000 
    if step % (train_args.save_interval * 2) == 0:
        try:
            embeddings = raw_model.tok_embeddings.weight[:proj_size].detach().cpu()
            metadata = []
            for i in range(proj_size):
                try:
                    token_str = tokenizer.decode([i]).replace('\n', '\\n').replace('\r', '')
                    metadata.append(token_str if token_str else f"<TOKEN_{i}>")
                except:
                    metadata.append(f"<UNK_{i}>")
            writer.add_embedding(embeddings, metadata=metadata, global_step=step, tag="TokenEmbeddings_Top2K")
        except Exception:
            pass

@torch.no_grad()
def log_text_generation(writer, model, tokenizer, device, step, prompt="人工智能", max_new_tokens=40):
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    raw_model.eval()
    try:
        input_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        x = torch.tensor([input_ids], dtype=torch.long, device=device)
        generated = []
        for _ in range(max_new_tokens):
            with torch.amp.autocast('cuda', dtype=torch.bfloat16): # 保持 BF16 一致性
                logits, _ = raw_model(x)
            next_token = torch.argmax(logits[0, -1, :]).item()
            if next_token == tokenizer.eos_id:
                break
            generated.append(next_token)
            x = torch.cat([x, torch.tensor([[next_token]], device=device)], dim=1)
        result_text = prompt + tokenizer.decode(generated)
        writer.add_text("Generation_Sample", result_text, step)
        print(f"\n[Sample] 🧠 生成测试: {result_text}\n")
    except Exception:
        pass
    finally:
        raw_model.train()

@torch.no_grad()
def evaluate(model, val_loader, device, eval_iters):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    val_iter = iter(val_loader)
    count = 0
    for _ in range(eval_iters):
        try:
            X, Y = next(val_iter)
        except StopIteration:
            break
        X, Y = X.to(device), Y.to(device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16): # 🌟 [极速优化 2] 使用 Bfloat16
            logits, loss = model(X, targets=Y)
        
        acc = calculate_accuracy(logits, Y)
        total_loss += loss.item()
        total_acc += acc
        count += 1
    model.train()
    return total_loss / max(count, 1), total_acc / max(count, 1)

# ================= 主训练循环 =================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_args = ModelArgs()
    train_args = TrainArgs()
    tokenizer = NaviTokenizer()

    resume_mode = os.path.exists(CHKPT_PATH)
    setup_directories(resume=resume_mode)
    writer = SummaryWriter(log_dir=LOGS_DIR)

    print(f"[Pre-train] 🚀 极速引擎已点火 | 设备: {device} (启用 TF32 & BF16 & Compile) | 步数: {train_args.total_steps:,}")
    
    full_dataset = load_from_disk(DATASET_PATH)
    split_ds = full_dataset.train_test_split(test_size=2000, seed=42)
    train_dataset = PackedIterableDataset(split_ds['train'], tokenizer, model_args.max_seq_len)
    val_dataset = PackedIterableDataset(split_ds['test'], tokenizer, model_args.max_seq_len)
    
    # 🌟 [极速优化 3]：提升 prefetch_factor 防止 GPU 饿肚子
    train_loader = DataLoader(train_dataset, batch_size=train_args.micro_batch_size, num_workers=4, pin_memory=True, prefetch_factor=4)
    val_loader = DataLoader(val_dataset, batch_size=train_args.micro_batch_size, num_workers=2)

    model = NaviLLM(model_args).to(device)
    
    # 🌟 [极速优化 4]：启动 torch.compile 图编译器，融合算子，巨幅提速并节省显存！
    print("[Pre-train] 正在进行 torch.compile 算子融合编译 (这可能需要几分钟，请耐心等待)...")
    model = torch.compile(model)
    print("[Pre-train] 编译完成，准备起飞！")

    # 🌟 [极速优化 5]：开启 Fused AdamW，合并显存更新核函数
    # 注意：获取参数需要从原始模型中拿，应对编译后的模型包裹
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=train_args.learning_rate,
                                   betas=(train_args.beta1, train_args.beta2),
                                   weight_decay=train_args.weight_decay,
                                   fused=True) 

    # 🌟 砸掉 GradScaler！由于使用了天然免疫下溢出的 BF16，缩放器反而成了拖慢 GPU 同步的累赘。

    start_step, best_val_loss = 0, float('inf')
    if resume_mode:
        checkpoint = torch.load(CHKPT_PATH, map_location=device)
        raw_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"[Pre-train] 成功接管断点，全局步: {start_step:,}")

    model.train()
    train_iter = iter(train_loader)
    global_step = start_step

    t0 = time.time()
    
    while global_step < train_args.total_steps:
        current_lr = get_lr(global_step, train_args)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        accumulated_acc = 0.0  

        for _ in range(train_args.gradient_accumulation_steps):
            try:
                X, Y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                X, Y = next(train_iter)
            X, Y = X.to(device), Y.to(device)

            # 🌟 [极速优化 2] 强制使用原生 Bfloat16，彻底消除溢出焦虑与缩放惩罚
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, loss = model(X, targets=Y)
                batch_acc = calculate_accuracy(logits, Y)
                accumulated_acc += batch_acc / train_args.gradient_accumulation_steps
                loss = loss / train_args.gradient_accumulation_steps

            # 没有 Scaler，直接硬核反传
            loss.backward()
            accumulated_loss += loss.item() * train_args.gradient_accumulation_steps

        # 没有 Scaler 的包裹，直接进行梯度裁剪和参数更新
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=train_args.grad_clip)
        optimizer.step()

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        tokens_per_sec = (train_args.gradient_accumulation_steps * train_args.micro_batch_size * model_args.max_seq_len) / dt

        # 高频监控记录
        if global_step % 10 == 0:
            # 记录平均 loss (而非 16 个微批次的总和)
            avg_loss = accumulated_loss / train_args.gradient_accumulation_steps
            log_metrics(writer, global_step, avg_loss, accumulated_acc, current_lr, grad_norm, tokens_per_sec)

        # 终端文本反馈
        if global_step % 100 == 0:
            avg_loss = accumulated_loss / train_args.gradient_accumulation_steps
            ppl = math.exp(min(avg_loss, 20))
            print(f"Step {global_step:07d}/{train_args.total_steps} | Loss: {avg_loss:.4f} | Acc: {accumulated_acc*100:.2f}% | PPL: {ppl:.1f} | LR: {current_lr:.2e} | Tok/s: {tokens_per_sec:.0f}")

        # 周期性全面体检
        if global_step > start_step and global_step % train_args.save_interval == 0:
            log_detailed_metrics(writer, model, tokenizer, global_step, train_args)
            log_text_generation(writer, model, tokenizer, device, global_step, prompt="在这片浩瀚的星空中，", max_new_tokens=40)
            
            torch.save({
                'step': global_step,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, CHKPT_PATH)
            
            val_loss, val_acc = evaluate(model, val_loader, device, train_args.eval_iters)
            val_ppl = math.exp(min(val_loss, 20))
            
            writer.add_scalar("Val/Loss", val_loss, global_step)
            writer.add_scalar("Val/Accuracy", val_acc, global_step)
            writer.add_scalar("Val/PPL", val_ppl, global_step)
            
            print(f"[Eval] 📊 验证完毕 | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val PPL: {val_ppl:.1f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(raw_model.state_dict(), BEST_PATH)
                print(f"[Eval] 👑 捕获到更优权重，已更新至 {BEST_PATH}")

        global_step += 1

    torch.save(raw_model.state_dict(), FINAL_PATH)
    writer.close()
    print(f"[Pre-train] 预训练全面结束。最终模型: {FINAL_PATH}")

if __name__ == "__main__":
    train()