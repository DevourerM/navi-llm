import os
import sys
import shutil
import math
import torch
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import load_from_disk
from dataclasses import dataclass

# ================= 路径与配置 =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from llm.llm_model import NaviLLM, ModelArgs
from dataset.token import NaviTokenizer

LOGS_DIR = os.path.join(BASE_DIR, "logs", "navi_llm")
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "skypile_15b")

CHKPT_PATH = os.path.join(MODEL_DIR, "checkpoint.pth")
BEST_PATH = os.path.join(MODEL_DIR, "best_model.pth")
FINAL_PATH = os.path.join(MODEL_DIR, "final_model.pth")

@dataclass
class TrainArgs:
    micro_batch_size: int = 4            
    gradient_accumulation_steps: int = 8 
    learning_rate: float = 3e-4          
    min_lr: float = 1e-5                 
    weight_decay: float = 0.1            
    beta1: float = 0.9                   
    beta2: float = 0.95                  
    grad_clip: float = 1.0               
    max_steps: int = 230000              
    warmup_steps: int = 5000             
    save_interval: int = 2000            
    eval_interval: int = 2000            
    eval_iters: int = 100                

# ================= 核心组件 =================
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
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y

def setup_directories():
    if os.path.exists(LOGS_DIR):
        print(f"🧹 清理旧日志: {LOGS_DIR}")
        shutil.rmtree(LOGS_DIR)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

def get_lr(step, args: TrainArgs):
    if step < args.warmup_steps:
        return args.learning_rate * (step + 1) / args.warmup_steps
    if step > args.max_steps:
        return args.min_lr
    decay_ratio = (step - args.warmup_steps) / (args.max_steps - args.warmup_steps)
    return args.min_lr + 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) * (args.learning_rate - args.min_lr)

# --- 🌟 高级可视化监控功能 ---
def log_advanced_metrics(writer, model, vocab_list, step, sample_size=2000):
    print(f"📸 抓取第 {step} 步的模型脑部 CT 并写入 TensorBoard...")
    # 1. 语义投影 (降采样前 2000 个高频词汇防止浏览器卡死)
    embedding_weights = model.tok_embeddings.weight.detach().cpu()
    writer.add_embedding(
        mat=embedding_weights[:sample_size], 
        metadata=vocab_list[:sample_size], 
        global_step=step,
        tag="Vocabulary_Latent_Space"
    )
    # 2. 关键参数梯度直方图 (安全脱离计算图)
    for name, param in model.named_parameters():
        if "layers.0" in name or "layers.15" in name or "tok_embeddings" in name:
            writer.add_histogram(f"Weights/{name}", param.detach().cpu(), step)
            if param.grad is not None:
                writer.add_histogram(f"Gradients/{name}", param.grad.detach().cpu(), step)

@torch.no_grad()
def evaluate(model, val_loader, device, eval_iters):
    model.eval()
    total_loss = 0.0
    val_iter = iter(val_loader)
    for _ in range(eval_iters):
        try: X, Y = next(val_iter)
        except StopIteration: break 
        X, Y = X.to(device), Y.to(device)
        with torch.cuda.amp.autocast():
            _, loss = model(X, targets=Y)
        total_loss += loss.item()
    model.train() 
    return total_loss / eval_iters

# ================= 主循环 =================
def train():
    setup_directories()
    writer = SummaryWriter(log_dir=LOGS_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 [Pre-train] 计算设备: {device}")
    
    model_args = ModelArgs()
    train_args = TrainArgs()
    tokenizer = NaviTokenizer()
    
    # 🌟 提取安全词表供可视化使用
    print("正在提取安全词表...")
    vocab_list = []
    for i in range(tokenizer.vocab_size):
        try:
            word = tokenizer.decode([i]).replace(" ", "").strip()
            vocab_list.append(word if word else f"<t_{i}>")
        except:
            vocab_list.append(f"<unk_{i}>")
            
    full_dataset = load_from_disk(DATASET_PATH)
    split_ds = full_dataset.train_test_split(test_size=2000, seed=42)
    train_dataset = PackedIterableDataset(split_ds['train'], tokenizer, model_args.max_seq_len)
    val_dataset = PackedIterableDataset(split_ds['test'], tokenizer, model_args.max_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=train_args.micro_batch_size, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=train_args.micro_batch_size, num_workers=1)

    model = NaviLLM(model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_args.learning_rate, weight_decay=train_args.weight_decay)
    scaler = torch.cuda.amp.GradScaler()

    start_step, best_val_loss = 0, float('inf')
    if os.path.exists(CHKPT_PATH):
        checkpoint = torch.load(CHKPT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"✅ 从第 {start_step} 步恢复训练")

    model.train()
    train_iter = iter(train_loader)
    
    for step in range(start_step, train_args.max_steps):
        lr = get_lr(step, train_args)
        for param_group in optimizer.param_groups: param_group['lr'] = lr
            
        try: X, Y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            X, Y = next(train_iter)
            
        X, Y = X.to(device), Y.to(device)
        
        with torch.cuda.amp.autocast():
            _, loss = model(X, targets=Y)
            loss = loss / train_args.gradient_accumulation_steps
            
        scaler.scale(loss).backward()
        
        if (step + 1) % train_args.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % 10 == 0:
            real_loss = loss.item() * train_args.gradient_accumulation_steps
            writer.add_scalar("Train/Loss", real_loss, step)
            writer.add_scalar("Train/LR", lr, step)
            print(f"Step {step:05d}/{train_args.max_steps} | Loss: {real_loss:.4f} | LR: {lr:.6f}")

        # 🌟 触发高级监控与保存断点
        if step % train_args.save_interval == 0 and step > start_step:
            log_advanced_metrics(writer, model, vocab_list, step)
            torch.save({'step': step, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_val_loss': best_val_loss}, CHKPT_PATH)

        if step % train_args.eval_interval == 0 and step > start_step:
            val_loss = evaluate(model, val_loader, device, train_args.eval_iters)
            writer.add_scalar("Val/Loss", val_loss, step)
            print(f"📊 评估结果: Val Loss = {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), BEST_PATH)

    torch.save(model.state_dict(), FINAL_PATH)
    writer.close()

if __name__ == "__main__":
    train()