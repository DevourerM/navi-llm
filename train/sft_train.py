import os
import sys
import json
import shutil
import math
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from llm.llm_model import NaviLLM, ModelArgs
from dataset.token import NaviTokenizer

SFT_DATA_PATH = os.path.join(BASE_DIR, "dataset", "navi_sft_data.jsonl")
SFT_LOGS_DIR = os.path.join(BASE_DIR, "logs", "sft_logs")
SFT_MODEL_DIR = os.path.join(BASE_DIR, "model", "sft_model")
PRETRAIN_PATH = os.path.join(BASE_DIR, "model", "best_model.pth") 
SFT_CHKPT_PATH = os.path.join(SFT_MODEL_DIR, "sft_checkpoint.pth")
SFT_FINAL_PATH = os.path.join(SFT_MODEL_DIR, "navi_sft_final.pth")

@dataclass
class SFTArgs:
    learning_rate: float = 2e-5          
    min_lr: float = 1e-6                 
    weight_decay: float = 0.05           
    micro_batch_size: int = 4            
    gradient_accumulation_steps: int = 4 
    epochs: int = 3                      
    warmup_steps: int = 100              
    max_seq_len: int = 1024              

# ================= 掩码数据集 =================
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

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        user_text = self.data[idx][0]["value"]
        navi_text = self.data[idx][1]["value"]
        
        prompt_str = f"User: {user_text}\nNavi: "
        reply_str = f"{navi_text}"
        
        prompt_tokens = self.tokenizer.encode(prompt_str, add_bos=True, add_eos=False)
        reply_tokens = self.tokenizer.encode(reply_str, add_bos=False, add_eos=True)
        
        input_ids = prompt_tokens + reply_tokens
        # 前文（用户提问部分）的 target 设置为 -100 忽略计算
        labels = [-100] * len(prompt_tokens) + reply_tokens
        
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            input_ids[-1], labels[-1] = self.tokenizer.eos_id, self.tokenizer.eos_id
            
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

def collate_fn(batch, pad_id=0):
    input_ids = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return input_ids_padded, labels_padded

def setup_directories():
    if os.path.exists(SFT_LOGS_DIR): shutil.rmtree(SFT_LOGS_DIR)
    os.makedirs(SFT_LOGS_DIR, exist_ok=True)
    os.makedirs(SFT_MODEL_DIR, exist_ok=True)

def get_lr(step, total_steps, args: SFTArgs):
    if step < args.warmup_steps: return args.learning_rate * (step + 1) / args.warmup_steps
    if step > total_steps: return args.min_lr
    decay_ratio = (step - args.warmup_steps) / (total_steps - args.warmup_steps)
    return args.min_lr + 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) * (args.learning_rate - args.min_lr)

# --- 🌟 高级可视化监控 ---
def log_advanced_metrics(writer, model, vocab_list, step, sample_size=2000):
    embedding_weights = model.tok_embeddings.weight.detach().cpu()
    writer.add_embedding(mat=embedding_weights[:sample_size], metadata=vocab_list[:sample_size], global_step=step, tag="Vocabulary_Latent_Space")
    for name, param in model.named_parameters():
        if "layers.0" in name or "layers.15" in name or "tok_embeddings" in name:
            writer.add_histogram(f"Weights/{name}", param.detach().cpu(), step)
            if param.grad is not None: writer.add_histogram(f"Gradients/{name}", param.grad.detach().cpu(), step)

# ================= 主循环 =================
def train():
    setup_directories()
    writer = SummaryWriter(log_dir=SFT_LOGS_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 [SFT 阶段] 计算设备: {device}")
    
    model_args = ModelArgs()
    sft_args = SFTArgs()
    tokenizer = NaviTokenizer()
    
    # 🌟 提取安全词表
    vocab_list = []
    for i in range(tokenizer.vocab_size):
        try:
            word = tokenizer.decode([i]).replace(" ", "").strip()
            vocab_list.append(word if word else f"<t_{i}>")
        except:
            vocab_list.append(f"<unk_{i}>")
            
    dataset = SFTDataset(SFT_DATA_PATH, tokenizer, sft_args.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=sft_args.micro_batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_id=tokenizer.pad_id))

    model = NaviLLM(model_args).to(device)
    if os.path.exists(SFT_CHKPT_PATH):
        checkpoint = torch.load(SFT_CHKPT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_step, start_epoch = checkpoint['step'], checkpoint['epoch']
    elif os.path.exists(PRETRAIN_PATH):
        model.load_state_dict(torch.load(PRETRAIN_PATH, map_location=device), strict=False)
        start_step, start_epoch = 0, 0
    else:
        raise FileNotFoundError(f"❌ 找不到预训练模型 {PRETRAIN_PATH}！")

    optimizer = torch.optim.AdamW(model.parameters(), lr=sft_args.learning_rate, weight_decay=sft_args.weight_decay)
    if 'checkpoint' in locals() and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    scaler = torch.cuda.amp.GradScaler()
    total_steps = (len(dataloader) // sft_args.gradient_accumulation_steps) * sft_args.epochs
    global_step = start_step

    model.train()
    for epoch in range(start_epoch, sft_args.epochs):
        for step_idx, (X, Y) in enumerate(dataloader):
            X, Y = X.to(device), Y.to(device)
            
            # ====== 🚨 核心修复：强制错位 (Shift) 🚨 ======
            X_shifted = X[:, :-1].contiguous()
            Y_shifted = Y[:, 1:].contiguous()
            # ==============================================
            
            lr = get_lr(global_step, total_steps, sft_args)
            for param_group in optimizer.param_groups: param_group['lr'] = lr
                
            with torch.cuda.amp.autocast():
                logits, loss = model(X_shifted, targets=Y_shifted)
                loss = loss / sft_args.gradient_accumulation_steps
                
            scaler.scale(loss).backward()
            
            if (step_idx + 1) % sft_args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 5 == 0:
                    real_loss = loss.item() * sft_args.gradient_accumulation_steps
                    writer.add_scalar("SFT/Train_Loss", real_loss, global_step)
                    writer.add_scalar("SFT/LR", lr, global_step)
                    print(f"[Epoch {epoch+1}] Step {global_step}/{total_steps} | Loss: {real_loss:.4f} | LR: {lr:.6f}")

                # 🌟 SFT 阶段通常比较短，每 200 步记录一次高级可视化并保存
                if global_step % 200 == 0:
                    log_advanced_metrics(writer, model, vocab_list, global_step)
                    torch.save({'epoch': epoch, 'step': global_step, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, SFT_CHKPT_PATH)

    torch.save(model.state_dict(), SFT_FINAL_PATH)
    writer.close()

if __name__ == "__main__":
    train()