import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class ModelArgs:
    """NaviLLM 模型核心超参数配置类"""
    vocab_size: int = 65024       # 词表大小
    max_seq_len: int = 2048       # 最大序列长度 (Context Window)
    dim: int = 1792               # 隐藏层维度 (Hidden Size) — 2048→1792, 降 15% 参数
    n_layers: int = 22            # Transformer 总层数 — 24→22
    n_heads: int = 14             # 注意力头数 — 16→14 (head_dim=128 不变)
    ffn_hidden_dim: int = 2304    # 加深变窄型 FFN 的内部隐藏层维度 — 2560→2304
    norm_eps: float = 1e-6        # RMSNorm 的稳定项 Epsilon
    dropout: float = 0.1          # Dropout 概率

class RMSNorm(nn.Module):
    """均方根归一化 (Root Mean Square Layer Normalization)"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)

def precompute_rope_freqs(dim: int, seq_len: int, theta: float = 10000.0):
    """预计算旋转位置编码 (RoPE) 的复数频率矩阵"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rope(xq, xk, freqs_cis):
    """将预计算的旋转位置编码应用到 Query 和 Key 张量上"""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2) 
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class SwiGLU(nn.Module):
    """经典门控线性单元激活层 (Swish-Gated Linear Unit)"""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class DeepResidualFFN(nn.Module):
    """加深变窄型前馈网络 (包含双层 SwiGLU 与内部残差直连)"""
    def __init__(self, dim: int, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.swiglu1 = SwiGLU(dim, hidden_dim)
        self.swiglu2 = SwiGLU(dim, hidden_dim)
        self.inner_norm = RMSNorm(dim, eps)

    def forward(self, x):
        h = self.swiglu1(x)
        h = x + h                    # 内部第一层残差直连
        h = self.inner_norm(h)
        h = self.swiglu2(h)
        return h

class CausalSelfAttention(nn.Module):
    """因果自注意力机制模块 (包含 QK-Norm 增强数值稳定性)"""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        
        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        self.wk = nn.Linear(args.dim, args.dim, bias=False)
        self.wv = nn.Linear(args.dim, args.dim, bias=False)
        self.wo = nn.Linear(args.dim, args.dim, bias=False)
        
        # QK-Norm: 稳定深层注意力机制的超大点积方差
        self.q_norm = RMSNorm(self.head_dim, eps=args.norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=args.norm_eps)
        
        self.dropout_p = args.dropout

    def forward(self, x, freqs_cis):
        B, T, C = x.shape
        
        xq = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        xk = self.wk(x).view(B, T, self.n_heads, self.head_dim)
        xv = self.wv(x).view(B, T, self.n_heads, self.head_dim)
        
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)
        
        xq, xk = apply_rope(xq, xk, freqs_cis)
        xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
        
        output = F.scaled_dot_product_attention(
            xq, xk, xv,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True
        )
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(output)

class TransformerBlock(nn.Module):
    """标准 Transformer 解码器块"""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = CausalSelfAttention(args)
        self.ffn = DeepResidualFFN(args.dim, args.ffn_hidden_dim, args.norm_eps)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cis):
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        out = h + self.ffn(self.ffn_norm(h))
        return out

class NaviLLM(nn.Module):
    """顶层根模型主体"""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        
        # 权重绑定 (Weight Tying)
        self.output.weight = self.tok_embeddings.weight
        self.freqs_cis = precompute_rope_freqs(args.dim // args.n_heads, args.max_seq_len)
        
        # 执行工业标准权值初始化
        self.apply(self._init_weights)
        self._init_residual_weights()

    def _init_weights(self, module):
        """基础权重分布初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _init_residual_weights(self):
        """🌟 核心修复：对所有残差输出层的投影矩阵应用深度缩放，锁死方差爆炸"""
        # 每个 Block 包含 3 个残差引入点 (Attention, SwiGLU1, SwiGLU2)
        scale_factor = 0.02 / math.sqrt(3 * self.args.n_layers)
        for name, param in self.named_parameters():
            if name.endswith('attention.wo.weight') or name.endswith('ffn.swiglu1.w2.weight') or name.endswith('ffn.swiglu2.w2.weight'):
                torch.nn.init.normal_(param, mean=0.0, std=scale_factor)

    def forward(self, tokens, targets=None):
        B, T = tokens.shape
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[:T].to(h.device)
        
        for layer in self.layers:
            h = layer(h, freqs_cis)
            
        h = self.norm(h)
        logits = self.output(h)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            
        return logits, loss