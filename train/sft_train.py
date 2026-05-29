import os
import sys
import json
import shutil
import math

# 必须在 import torch 前设置，允许动态扩展显存段
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass

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
    # --- 批次与轮数 ---
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 4     # 等效 batch_size = 16
    epochs: int = 3                           # 35k 数据, 3 epoch 避免过拟合
    max_seq_len: int = 1024

    # --- 学习率 ---
    learning_rate: float = 1e-5               # SFT 用更低 lr, 保护预训练知识
    min_lr: float = 1e-6
    warmup_steps: int = 150                   # 占总步数约 1.5%

    # --- 优化器 ---
    weight_decay: float = 0.1
    grad_clip: float = 1.0

# ================= SFT 数据集 =================
class SFTDataset(Dataset):
    """SFT 对话数据集: prompt 部分用 -100 mask, 仅对回复计算 loss"""

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
        labels = [-100] * len(prompt_tokens) + reply_tokens  # prompt 不参与 loss

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            input_ids[-1] = self.tokenizer.eos_id
            labels[-1] = self.tokenizer.eos_id

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def collate_fn(batch, pad_id=0):
    """批次填充: input 填 pad_id, labels 填 -100 (cross_entropy 忽略)"""
    input_ids = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return input_ids_padded, labels_padded

# ================= 工具函数 =================
def setup_directories():
    """初始化日志和模型目录 (从头训练时清空旧日志)"""
    if os.path.exists(SFT_LOGS_DIR):
        shutil.rmtree(SFT_LOGS_DIR)
    os.makedirs(SFT_LOGS_DIR, exist_ok=True)
    os.makedirs(SFT_MODEL_DIR, exist_ok=True)


def get_lr(step, total_steps, args: SFTArgs):
    """余弦退火学习率调度"""
    if step < args.warmup_steps:
        return args.learning_rate * (step + 1) / args.warmup_steps
    if step > total_steps:
        return args.min_lr
    decay_ratio = (step - args.warmup_steps) / (total_steps - args.warmup_steps)
    return args.min_lr + 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) * (args.learning_rate - args.min_lr)


def log_metrics(writer, model, global_step, loss, lr, epoch):
    """记录 SFT 训练指标到 TensorBoard"""
    # 标量
    writer.add_scalar("SFT/Train_Loss", loss, global_step)
    writer.add_scalar("SFT/LR", lr, global_step)
    writer.add_scalar("SFT/Epoch", epoch, global_step)

    # 权重直方图 (每 500 优化步记录一次, 避免 I/O 过重)
    if global_step % 500 == 0:
        for name, param in model.named_parameters():
            if "layers.0" in name or f"layers.{model.args.n_layers - 1}" in name or "tok_embeddings" in name:
                writer.add_histogram(f"SFT_Weights/{name}", param.detach().cpu(), global_step)
                if param.grad is not None:
                    writer.add_histogram(f"SFT_Gradients/{name}", param.grad.detach().cpu(), global_step)

# ================= 主训练循环 =================
def train():
    setup_directories()
    writer = SummaryWriter(log_dir=SFT_LOGS_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SFT] 计算设备: {device}")

    model_args = ModelArgs()
    sft_args = SFTArgs()
    tokenizer = NaviTokenizer()

    # 加载数据
    dataset = SFTDataset(SFT_DATA_PATH, tokenizer, sft_args.max_seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=sft_args.micro_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id=tokenizer.pad_id)
    )
    print(f"[SFT] 数据集: {len(dataset)} 条, {sft_args.epochs} epochs")

    # 加载模型
    model = NaviLLM(model_args).to(device)
    if os.path.exists(SFT_CHKPT_PATH):
        checkpoint = torch.load(SFT_CHKPT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_step, start_epoch = checkpoint['step'], checkpoint['epoch']
        print(f"[SFT] 从 SFT checkpoint 恢复: epoch {start_epoch}, step {start_step}")
    elif os.path.exists(PRETRAIN_PATH):
        model.load_state_dict(torch.load(PRETRAIN_PATH, map_location=device), strict=False)
        start_step, start_epoch = 0, 0
        print(f"[SFT] 加载预训练权重: {PRETRAIN_PATH}")
    else:
        raise FileNotFoundError(f"预训练模型不存在: {PRETRAIN_PATH}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[SFT] 参数总量: {total_params / 1e9:.2f}B")

    optimizer = torch.optim.AdamW(model.parameters(), lr=sft_args.learning_rate,
                                   weight_decay=sft_args.weight_decay)
    if 'checkpoint' in locals() and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    scaler = torch.amp.GradScaler('cuda')
    steps_per_epoch = len(dataloader) // sft_args.gradient_accumulation_steps
    total_steps = steps_per_epoch * sft_args.epochs
    global_step = start_step

    model.train()
    for epoch in range(start_epoch, sft_args.epochs):
        epoch_loss = 0.0
        epoch_batches = 0

        for step_idx, (X, Y) in enumerate(dataloader):
            X, Y = X.to(device), Y.to(device)

            # Teacher Forcing: 输入右移一位, 目标也右移一位
            X_shifted = X[:, :-1].contiguous()
            Y_shifted = Y[:, 1:].contiguous()

            lr = get_lr(global_step, total_steps, sft_args)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            with torch.amp.autocast('cuda'):
                logits, loss = model(X_shifted, targets=Y_shifted)
                loss = loss / sft_args.gradient_accumulation_steps

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * sft_args.gradient_accumulation_steps
            epoch_batches += 1

            # 梯度累积步结束
            if (step_idx + 1) % sft_args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=sft_args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                # 每 10 个优化步记录一次
                if global_step % 10 == 0:
                    avg_loss = epoch_loss / max(epoch_batches, 1)
                    log_metrics(writer, model, global_step, avg_loss, lr, epoch + 1)

                    ppl = math.exp(min(avg_loss, 20))
                    print(f"[Epoch {epoch+1}/{sft_args.epochs}] Step {global_step}/{total_steps} | Loss: {avg_loss:.4f} | PPL: {ppl:.1f} | LR: {lr:.2e}")

                # 存档
                if global_step % 400 == 0:
                    torch.save({
                        'epoch': epoch,
                        'step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, SFT_CHKPT_PATH)

        # Epoch 结束统计
        avg_epoch_loss = epoch_loss / max(epoch_batches, 1)
        epoch_ppl = math.exp(min(avg_epoch_loss, 20))
        writer.add_scalar("SFT/Epoch_Loss", avg_epoch_loss, epoch)
        writer.add_scalar("SFT/Epoch_PPL", epoch_ppl, epoch)
        print(f"[Epoch {epoch+1}/{sft_args.epochs}] 完成 | Avg Loss: {avg_epoch_loss:.4f} | PPL: {epoch_ppl:.1f}")

    # 训练结束
    torch.save(model.state_dict(), SFT_FINAL_PATH)
    writer.add_hparams(
        {k: getattr(sft_args, k) for k in ['learning_rate', 'epochs', 'weight_decay']},
        {'final_epoch_loss': avg_epoch_loss},
    )
    writer.close()
    print(f"[SFT] 训练完成! 最终模型: {SFT_FINAL_PATH}")


if __name__ == "__main__":
    train()