import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# ==========================================
# 1. 超参数配置类
# ==========================================
@dataclass
class ModelArgs:
    # 基础参数
    vocab_size: int = 65024     
    max_seq_len: int = 2048     
    dim: int = 2048             
    n_layers: int = 24         
    n_heads: int = 16           
    
    # 加深变窄 FFN
    ffn_hidden_dim: int = 2560  
    
    # 其他参数
    norm_eps: float = 1e-6      
    dropout: float = 0.1

# ==========================================
# 2. 核心基础组件
# ==========================================
class RMSNorm(nn.Module):
    """均方根归一化"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)

def precompute_rope_freqs(dim: int, seq_len: int, theta: float = 10000.0):
    """预计算 RoPE 旋转频率矩阵"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rope(xq, xk, freqs_cis):
    """应用旋转位置编码"""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2) 
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# ==========================================
# 3. 加深变窄 FFN + SwiGLU
# ==========================================
class SwiGLU(nn.Module):
    """单层 SwiGLU 门控激活"""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class DeepResidualFFN(nn.Module):
    """加深变窄 FFN: 两层 SwiGLU + 残差直连
    
       x → SwiGLU₁ → +x (残差) → Norm → SwiGLU₂ → 输出
       
    参数量: 2 × 3 × dim × hidden_dim ≈ 2 × 3 × 2048 × 4096 ≈ 50M/层
    """
    def __init__(self, dim: int, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.swiglu1 = SwiGLU(dim, hidden_dim)
        self.swiglu2 = SwiGLU(dim, hidden_dim)
        self.inner_norm = RMSNorm(dim, eps)

    def forward(self, x):
        h = self.swiglu1(x)
        h = x + h                    # 🌟 残差直连
        h = self.inner_norm(h)
        h = self.swiglu2(h)
        return h

# ==========================================
# 4. 上下文压缩器 + 注意力机制
# ==========================================
class CausalSelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        
        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        self.wk = nn.Linear(args.dim, args.dim, bias=False)
        self.wv = nn.Linear(args.dim, args.dim, bias=False)
        self.wo = nn.Linear(args.dim, args.dim, bias=False)
        
        # QK-Norm
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

# ==========================================
# 5. Transformer 块
# ==========================================
class TransformerBlock(nn.Module):
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

# ==========================================
# 6. 顶层模型：NaviLLM
# ==========================================
class NaviLLM(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        
        # 权重绑定
        self.output.weight = self.tok_embeddings.weight
        self.freqs_cis = precompute_rope_freqs(args.dim // args.n_heads, args.max_seq_len)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """标准正态分布初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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