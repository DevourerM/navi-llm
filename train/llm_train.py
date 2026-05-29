import os
import sys
import shutil
import math

# 必须在 import torch 前设置，允许动态扩展显存段，消除碎片化 OOM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch

from torch.utils.data import IterableDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import load_from_disk
from dataclasses import dataclass

# ================= 路径配置 =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from llm.llm_model import NaviLLM, ModelArgs
from dataset.navi_token import NaviTokenizer

LOGS_DIR = os.path.join(BASE_DIR, "logs", "navi_llm")
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "skypile_100b")  # ~100B tokens

CHKPT_PATH = os.path.join(MODEL_DIR, "checkpoint.pth")
BEST_PATH = os.path.join(MODEL_DIR, "best_model.pth")
FINAL_PATH = os.path.join(MODEL_DIR, "final_model.pth")

# ================= 训练超参数 =================
@dataclass
class TrainArgs:
    # --- 批次与步数 ---
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 16    # 等效 batch_size = 32
    total_steps: int = 1500000               # 总步数 (~98B tokens, ~75 tokens/param)
    cosine_steps: int = 900000               # 余弦退火终点 (60% total, 到达后恒定 min_lr)
    warmup_steps: int = 7500                 # warmup 占 cosine 的 0.8%

    # --- 学习率 ---
    learning_rate: float = 3e-4              # 峰值学习率 (1.3B 模型标准)
    min_lr: float = 3e-6                     # 最小学习率 (峰值的 1/100, 稠密模型可更低)
    final_lr: float = 1e-6                   # 最终恒定 lr (余弦结束后的持续阶段)

    # --- 优化器 ---
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # --- 日志与存档 ---
    save_interval: int = 20000               # 每 2 万步存档 (~1.3B tokens 间隔)
    eval_interval: int = 20000
    eval_iters: int = 100

# ================= 数据集 =================
class PackedIterableDataset(IterableDataset):
    """连续打包数据集: 将变长文档拼接为固定长度序列, 无 padding 浪费"""

    def __init__(self, hf_dataset, tokenizer, seq_len):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self):
        buffer = []
        for item in self.dataset:
            text = item.get("text", "").strip()
            if not text:
                continue
            tokens = self.tokenizer.encode(text, add_bos=True, add_eos=True)
            buffer.extend(tokens)

            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len + 1:]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y

# ================= 工具函数 =================
def setup_directories():
    """初始化日志和模型目录 (从头训练时清空旧日志)"""
    if os.path.exists(LOGS_DIR):
        print(f"[Setup] 清理旧日志: {LOGS_DIR}")
        shutil.rmtree(LOGS_DIR)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)


def get_lr(step, args: TrainArgs):
    """三阶段学习率调度:
      1) warmup:        0 -> learning_rate (线性)
      2) cosine decay:  learning_rate -> min_lr
      3) constant:      final_lr (余弦结束后保持, 用于长程续训)
    """
    if step < args.warmup_steps:
        return args.learning_rate * (step + 1) / args.warmup_steps
    if step >= args.cosine_steps:
        return args.final_lr
    decay_ratio = (step - args.warmup_steps) / (args.cosine_steps - args.warmup_steps)
    return args.min_lr + 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) * (args.learning_rate - args.min_lr)


def log_metrics(writer, model, step, loss, lr, grad_norm=None):
    """记录训练指标到 TensorBoard — 轻量级版本, 每 10 步调用"""

    # 标量指标
    writer.add_scalar("Train/Loss", loss, step)
    writer.add_scalar("Train/LR", lr, step)
    if grad_norm is not None:
        writer.add_scalar("Train/GradNorm", grad_norm, step)


def log_detailed_metrics(writer, model, step, train_args, tokenizer=None):
    """记录详细诊断指标 — 每 save_interval 调用一次"""

    # 1. 权重与梯度直方图 (首层、末层、embedding)
    for name, param in model.named_parameters():
        if "layers.0" in name or f"layers.{model.args.n_layers - 1}" in name or "tok_embeddings" in name:
            writer.add_histogram(f"Weights/{name}", param.detach().cpu(), step)
            if param.grad is not None:
                writer.add_histogram(f"Gradients/{name}", param.grad.detach().cpu(), step)

    # 2. 参数范数 (监控数值稳定性)
    total_norm = 0.0
    for p in model.parameters():
        if p is not None:
            total_norm += p.norm(2).item() ** 2
    writer.add_scalar("Debug/ParamNorm", total_norm ** 0.5, step)

    # 3. 模型架构图 (首次写入)
    if step <= train_args.save_interval:
        try:
            dummy = torch.randint(0, 1000, (1, 64)).to(next(model.parameters()).device)
            writer.add_graph(model, dummy)
        except Exception:
            pass  # add_graph 在某些 PyTorch 版本可能失败, 不影响训练


@torch.no_grad()
def evaluate(model, val_loader, device, eval_iters):
    """验证集上评估 perplexity"""
    model.eval()
    total_loss = 0.0
    val_iter = iter(val_loader)
    count = 0
    for _ in range(eval_iters):
        try:
            X, Y = next(val_iter)
        except StopIteration:
            break
        X, Y = X.to(device), Y.to(device)
        with torch.amp.autocast('cuda'):
            _, loss = model(X, targets=Y)
        total_loss += loss.item()
        count += 1
    model.train()
    return total_loss / max(count, 1)


# ================= 主训练循环 =================
def train():
    setup_directories()
    writer = SummaryWriter(log_dir=LOGS_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Pre-train] 计算设备: {device}")
    print(f"[Pre-train] 总步数: {TrainArgs.total_steps:,} (~98B tokens)")
    print(f"[Pre-train] 模型架构: 24层 / dim=2048 / 16头 / DeepResidualFFN+QK-Norm / ~1.3B 参数")

    model_args = ModelArgs()
    train_args = TrainArgs()
    tokenizer = NaviTokenizer()

    # 加载数据集
    full_dataset = load_from_disk(DATASET_PATH)
    split_ds = full_dataset.train_test_split(test_size=2000, seed=42)
    train_dataset = PackedIterableDataset(split_ds['train'], tokenizer, model_args.max_seq_len)
    val_dataset = PackedIterableDataset(split_ds['test'], tokenizer, model_args.max_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=train_args.micro_batch_size, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=train_args.micro_batch_size, num_workers=1)

    model = NaviLLM(model_args).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Pre-train] 参数总量: {total_params / 1e9:.2f}B")

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_args.learning_rate,
                                   weight_decay=train_args.weight_decay,
                                   betas=(train_args.beta1, train_args.beta2))
    scaler = torch.amp.GradScaler('cuda')

    # 恢复训练
    start_step, best_val_loss = 0, float('inf')
    if os.path.exists(CHKPT_PATH):
        checkpoint = torch.load(CHKPT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        scaler_state = checkpoint.get('scaler_state_dict')
        if scaler_state:
            scaler.load_state_dict(scaler_state)
        print(f"[Pre-train] 从第 {start_step:,} 步恢复训练, best_val_loss={best_val_loss:.4f}")

    model.train()
    train_iter = iter(train_loader)
    global_step = start_step  # 独立计数器, 用于日志 (避免 step 与累积步混淆)

    for step in range(start_step, train_args.total_steps):
        # 学习率调度
        lr = get_lr(step, train_args)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 数据加载
        try:
            X, Y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            X, Y = next(train_iter)
        X, Y = X.to(device), Y.to(device)

        # 前向 + 反向
        with torch.amp.autocast('cuda'):
            _, loss = model(X, targets=Y)
            loss = loss / train_args.gradient_accumulation_steps

        scaler.scale(loss).backward()

        # 梯度累积步结束后更新参数
        if (step + 1) % train_args.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        # 高频日志 (每 10 步)
        if step % 10 == 0:
            real_loss = loss.item() * train_args.gradient_accumulation_steps
            log_metrics(writer, model, step, real_loss, lr)

        # 低频详细日志 (每 100 步打印)
        if step % 100 == 0:
            real_loss = loss.item() * train_args.gradient_accumulation_steps
            ppl = math.exp(min(real_loss, 20))
            print(f"Step {step:07d}/{train_args.total_steps} | Loss: {real_loss:.4f} | PPL: {ppl:.1f} | LR: {lr:.2e}")

        # 存档与详细诊断
        if step % train_args.save_interval == 0 and step > start_step:
            log_detailed_metrics(writer, model, step, train_args, tokenizer)
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_loss': best_val_loss,
            }, CHKPT_PATH)
            print(f"[Checkpoint] 已保存至 {CHKPT_PATH}")

        # 验证
        if step % train_args.eval_interval == 0 and step > start_step:
            val_loss = evaluate(model, val_loader, device, train_args.eval_iters)
            val_ppl = math.exp(min(val_loss, 20))
            writer.add_scalar("Val/Loss", val_loss, step)
            writer.add_scalar("Val/PPL", val_ppl, step)
            print(f"[Eval] Step {step} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.1f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), BEST_PATH)
                print(f"[Eval] 新的最佳模型! Val Loss: {val_loss:.4f}")

    # 训练结束
    torch.save(model.state_dict(), FINAL_PATH)
    writer.add_hparams(
        {k: getattr(train_args, k) for k in ['learning_rate', 'total_steps', 'weight_decay']},
        {'final_val_loss': best_val_loss},
    )
    writer.close()
    print(f"[Pre-train] 训练完成! 最终模型: {FINAL_PATH}")


if __name__ == "__main__":
    train()