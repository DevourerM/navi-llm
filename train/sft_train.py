import os
import sys
import json
import shutil
import math
import time

# 优化 CUDA 显存管理器
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch

# 🌟 [极速优化 1]：开启 TF32，彻底释放 5090 的 Tensor Core 矩阵算力
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass

# ================= 路径配置 =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from llm.llm_model import NaviLLM, ModelArgs
from dataset.navi_token import NaviTokenizer

SFT_DATA_PATH = os.path.join(BASE_DIR, "dataset", "navi_sft_data.jsonl")
SFT_LOGS_DIR = os.path.join(BASE_DIR, "logs", "sft_logs")
SFT_MODEL_DIR = os.path.join(BASE_DIR, "model", "sft_model")
PRETRAIN_PATH = os.path.join(BASE_DIR, "model", "best_model.pth")
SFT_CHKPT_PATH = os.path.join(SFT_MODEL_DIR, "sft_checkpoint.pth")
SFT_FINAL_PATH = os.path.join(SFT_MODEL_DIR, "navi_sft_final.pth")

# ================= SFT 训练超参数 =================
@dataclass
class SFTArgs:
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 4     
    epochs: int = 3                          
    max_seq_len: int = 1024                  

    learning_rate: float = 1e-5              
    min_lr: float = 1e-6
    warmup_steps: int = 150                  

    weight_decay: float = 0.1
    grad_clip: float = 1.0

# ================= SFT 数据集 =================
class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                if "conversations" in item and len(item["conversations"]) >= 2:
                    self.data.append(item["conversations"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user_text = self.data[idx][0]["value"]
        navi_text = self.data[idx][1]["value"]

        prompt_str = f"User: {user_text}\nNavi: "
        reply_str = f"{navi_text}"

        prompt_tokens = self.tokenizer.encode(prompt_str, add_bos=True, add_eos=False)
        reply_tokens = self.tokenizer.encode(reply_str, add_bos=False, add_eos=True)

        input_ids = prompt_tokens + reply_tokens
        labels = [-100] * len(prompt_tokens) + reply_tokens  

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            input_ids[-1] = self.tokenizer.eos_id
            labels[-1] = self.tokenizer.eos_id

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

def collate_fn(batch, pad_id=0):
    input_ids = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    
    return input_ids_padded, labels_padded

# ================= 工具函数 =================
def setup_directories(resume=False):
    if not resume and os.path.exists(SFT_LOGS_DIR):
        print(f"[Setup] 清理旧日志，执行干净初始化: {SFT_LOGS_DIR}")
        shutil.rmtree(SFT_LOGS_DIR)
    os.makedirs(SFT_LOGS_DIR, exist_ok=True)
    os.makedirs(SFT_MODEL_DIR, exist_ok=True)

def get_lr(step, total_steps, args: SFTArgs):
    if step < args.warmup_steps:
        return args.learning_rate * (step + 1) / args.warmup_steps
    if step >= total_steps:
        return args.min_lr
    decay_ratio = (step - args.warmup_steps) / (total_steps - args.warmup_steps)
    return args.min_lr + 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) * (args.learning_rate - args.min_lr)

def calculate_accuracy(logits, targets, ignore_index=-100):
    preds = torch.argmax(logits, dim=-1)
    valid_mask = (targets != ignore_index)
    correct = (preds == targets) & valid_mask
    acc = correct.sum().float() / torch.clamp(valid_mask.sum().float(), min=1.0)
    return acc.item()

def log_metrics(writer, global_step, loss, acc, lr, epoch, throughput):
    writer.add_scalar("SFT/Train_Loss", loss, global_step)
    writer.add_scalar("SFT/Accuracy", acc, global_step)
    writer.add_scalar("SFT/LR", lr, global_step)
    writer.add_scalar("SFT/Epoch", epoch, global_step)
    writer.add_scalar("Perf/Tokens_per_Sec", throughput, global_step)

def log_detailed_metrics(writer, model, tokenizer, step):
    # 提取底层原始模型
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    for name, param in raw_model.named_parameters():
        if "layers.0" in name or f"layers.{raw_model.args.n_layers - 1}" in name or "tok_embeddings" in name:
            writer.add_histogram(f"SFT_Weights/{name}", param.detach().cpu(), step)
            if param.grad is not None:
                writer.add_histogram(f"SFT_Gradients/{name}", param.grad.detach().cpu(), step)
                
    proj_size = 2000 
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
def log_text_generation(writer, model, tokenizer, device, step, prompt="你是谁？", max_new_tokens=60):
    # 🌟 [极速优化 避坑]：生成任务绝对不能用 compile 后的模型，否则每次循环都会触发重编译！
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    raw_model.eval()
    try:
        formatted_prompt = f"User: {prompt}\nNavi: "
        input_ids = tokenizer.encode(formatted_prompt, add_bos=True, add_eos=False)
        x = torch.tensor([input_ids], dtype=torch.long, device=device)
        generated = []
        
        for _ in range(max_new_tokens):
            with torch.amp.autocast('cuda', dtype=torch.bfloat16): # 保持 BF16
                logits, _ = raw_model(x)
            next_token = torch.argmax(logits[0, -1, :]).item()
            if next_token == tokenizer.eos_id:
                break
            generated.append(next_token)
            x = torch.cat([x, torch.tensor([[next_token]], device=device)], dim=1)
            
        result_text = tokenizer.decode(generated)
        writer.add_text("SFT_Generation", f"**Q:** {prompt}\n\n**Navi:** {result_text}", step)
        print(f"\n[SFT Chat Sample] 🧠 \nUser: {prompt}\nNavi: {result_text}\n")
    except Exception:
        pass
    finally:
        raw_model.train()

# ================= 主训练循环 =================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_args = ModelArgs()
    sft_args = SFTArgs()
    tokenizer = NaviTokenizer()

    resume_mode = os.path.exists(SFT_CHKPT_PATH)
    setup_directories(resume=resume_mode)
    writer = SummaryWriter(log_dir=SFT_LOGS_DIR)

    print(f"[SFT] 🚀 极速指令微调引擎已点火 | 设备: {device} (TF32+BF16+Compile)")

    dataset = SFTDataset(SFT_DATA_PATH, tokenizer, sft_args.max_seq_len)
    
    # 🌟 [极速优化 2]：满载预取，榨干 CPU 瓶颈
    dataloader = DataLoader(
        dataset,
        batch_size=sft_args.micro_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=4,
        collate_fn=lambda b: collate_fn(b, pad_id=tokenizer.pad_id)
    )
    
    steps_per_epoch = len(dataloader) // sft_args.gradient_accumulation_steps
    total_steps = steps_per_epoch * sft_args.epochs
    print(f"[SFT] 数据集: {len(dataset)} 条 | Epochs: {sft_args.epochs} | Total Steps: {total_steps}")

    model = NaviLLM(model_args).to(device)
    
    start_step, start_epoch = 0, 0
    if resume_mode:
        checkpoint = torch.load(SFT_CHKPT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_step, start_epoch = checkpoint['step'], checkpoint['epoch']
        print(f"[SFT] 从 SFT checkpoint 恢复: epoch {start_epoch}, step {start_step}")
    elif os.path.exists(PRETRAIN_PATH):
        model.load_state_dict(torch.load(PRETRAIN_PATH, map_location=device), strict=False)
        print(f"[SFT] 成功继承预训练权重: {PRETRAIN_PATH}")
    else:
        raise FileNotFoundError(f"未找到预训练模型基础权重: {PRETRAIN_PATH}")

    # 🌟 [极速优化 3]：挂载图编译。开启 dynamic=True 容忍 SFT 数据的不定长 Padding
    print("[SFT] 正在进行 torch.compile (动态形状适配模式)，请稍候...")
    model = torch.compile(model, dynamic=True)
    print("[SFT] 编译完成，准备起飞！")

    # 🌟 [极速优化 4]：Fused AdamW
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=sft_args.learning_rate, weight_decay=sft_args.weight_decay, fused=True)
    
    if resume_mode and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 砸掉 GradScaler，全面依赖 Bfloat16
    model.train()
    global_step = start_step
    t0 = time.time()

    for epoch in range(start_epoch, sft_args.epochs):
        epoch_loss = 0.0
        epoch_global_steps = 0
        accumulated_loss = 0.0 
        accumulated_acc = 0.0

        for step_idx, (X, Y) in enumerate(dataloader):
            X, Y = X.to(device), Y.to(device)

            X_shifted = X[:, :-1].contiguous()
            Y_shifted = Y[:, 1:].contiguous()

            current_lr = get_lr(global_step, total_steps, sft_args)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            # 🌟 [极速优化 5]：纯正 BF16 向前传播
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, loss = model(X_shifted, targets=Y_shifted)
                
                batch_acc = calculate_accuracy(logits, Y_shifted)
                accumulated_acc += batch_acc / sft_args.gradient_accumulation_steps
                
                loss = loss / sft_args.gradient_accumulation_steps

            loss.backward()
            accumulated_loss += loss.item() * sft_args.gradient_accumulation_steps

            if (step_idx + 1) % sft_args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=sft_args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
                global_step += 1
                epoch_loss += accumulated_loss
                epoch_global_steps += 1
                
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                tokens_per_sec = (sft_args.gradient_accumulation_steps * sft_args.micro_batch_size * sft_args.max_seq_len) / dt

                if global_step % 10 == 0:
                    avg_loss = accumulated_loss / sft_args.gradient_accumulation_steps
                    log_metrics(writer, global_step, avg_loss, accumulated_acc, current_lr, epoch + 1, tokens_per_sec)

                if global_step % 50 == 0:
                    avg_loss = accumulated_loss / sft_args.gradient_accumulation_steps
                    ppl = math.exp(min(avg_loss, 20))
                    print(f"[Epoch {epoch+1}/{sft_args.epochs}] Step {global_step}/{total_steps} | Loss: {avg_loss:.4f} | Acc: {accumulated_acc*100:.2f}% | PPL: {ppl:.1f} | LR: {current_lr:.2e} | Tok/s: {tokens_per_sec:.0f}")
                
                if global_step % 400 == 0:
                    log_detailed_metrics(writer, model, tokenizer, global_step)
                    log_text_generation(writer, model, tokenizer, device, global_step, prompt="你是谁？")
                    
                    torch.save({
                        'epoch': epoch,
                        'step': global_step,
                        'model_state_dict': raw_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, SFT_CHKPT_PATH)

                accumulated_loss = 0.0
                accumulated_acc = 0.0

        avg_epoch_loss = epoch_loss / max(epoch_global_steps, 1)
        epoch_ppl = math.exp(min(avg_epoch_loss, 20))
        writer.add_scalar("SFT/Epoch_Avg_Loss", avg_epoch_loss, epoch + 1)
        print(f"✅ [Epoch {epoch+1}/{sft_args.epochs}] 圆满完成 | Avg Loss: {avg_epoch_loss:.4f} | PPL: {epoch_ppl:.1f}")

    torch.save(raw_model.state_dict(), SFT_FINAL_PATH)
    writer.close()
    print(f"🎉 [SFT] Navi 的灵魂注入完成! 最终模型已封存至: {SFT_FINAL_PATH}")

if __name__ == "__main__":
    train()