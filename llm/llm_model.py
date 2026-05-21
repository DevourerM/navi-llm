import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# ==========================================
# 1. 超参数配置类 (支持直接对接 config.json)
# ==========================================
@dataclass
class ModelArgs:
    # 基础参数
    vocab_size: int = 65024     
    max_seq_len: int = 2048     
    dim: int = 2048             
    n_layers: int = 16          
    n_heads: int = 16           
    
    # DeepSeekMoE 专属参数
    moe_shared_dim: int = 2048  
    moe_routed_dim: int = 384   
    moe_n_experts: int = 16     
    moe_top_k: int = 4          
    
    # 其他参数
    norm_eps: float = 1e-6      
    dropout: float = 0.1

# ==========================================
# 2. 核心基础组件
# ==========================================
class RMSNorm(nn.Module):
    """均方根归一化 (替代传统 LayerNorm)"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)

def precompute_rope_freqs(dim: int, seq_len: int, theta: float = 10000.0):
    """预计算 RoPE 的旋转频率矩阵"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rope(xq, xk, freqs_cis):
    """应用旋转位置编码 (强制 float32 保证混合精度下的数学稳定性)"""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2) # [1, seq_len, 1, dim/2]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# ==========================================
# 3. 激活函数与专家网络
# ==========================================
class SwiGLU(nn.Module):
    """SwiGLU 激活网络 (大模型非线性表达的核心)"""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class DeepSeekMoE(nn.Module):
    """DeepSeekMoE: 共享专家 + 细粒度路由专家"""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.moe_top_k
        self.num_experts = args.moe_n_experts
        
        self.shared_expert = SwiGLU(args.dim, args.moe_shared_dim)
        self.routed_experts = nn.ModuleList([
            SwiGLU(args.dim, args.moe_routed_dim) for _ in range(self.num_experts)
        ])
        self.router = nn.Linear(args.dim, self.num_experts, bias=False)

    def forward(self, x):
        shared_out = self.shared_expert(x)
        
        router_logits = self.router(x)
        routing_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        # 提取被选中的专家输出并加权
        all_expert_outputs = torch.stack([expert(x) for expert in self.routed_experts], dim=-2) 
        B, T, _ = x.shape
        routed_out = torch.zeros_like(x)
        
        for k in range(self.top_k):
            exp_idx = selected_experts[..., k]       
            weight = routing_weights[..., k]         
            for b in range(B):
                for t in range(T):
                    idx = exp_idx[b, t]
                    routed_out[b, t] += weight[b, t] * all_expert_outputs[b, t, idx]
                    
        return shared_out + routed_out

# ==========================================
# 4. 注意力机制与 Transformer 块
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
        self.dropout_p = args.dropout

    def forward(self, x, freqs_cis):
        B, T, C = x.shape
        
        xq = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        xk = self.wk(x).view(B, T, self.n_heads, self.head_dim)
        xv = self.wv(x).view(B, T, self.n_heads, self.head_dim)
        
        xq, xk = apply_rope(xq, xk, freqs_cis)
        
        # 形状变换以适配 SDPA: [B, n_heads, T, head_dim]
        xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
        
        # 🚀 核心优化：使用 PyTorch 2.0+ 内置的 FlashAttention (SDPA)
        # is_causal=True 会自动在底层生成并应用因果掩码，速度极快！
        output = F.scaled_dot_product_attention(
            xq, xk, xv,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True
        )
        
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(output)

class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = CausalSelfAttention(args)
        self.ffn = DeepSeekMoE(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cis):
        # 移除了手动的 mask，让 SDPA 自动处理
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        out = h + self.ffn(self.ffn_norm(h))
        return out

# ==========================================
# 5. 顶层模型：NaviLLM
# ==========================================
class NaviLLM(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        
        self.layers = nn.ModuleList([
            TransformerBlock(args) for _ in range(args.n_layers)
        ])
        
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        
        # 🔗 核心优化：权重绑定 (Weight Tying)
        # 让输出层的权重直接使用 Embedding 层的权重，大幅减少参数量并加速收敛
        self.output.weight = self.tok_embeddings.weight
        
        self.freqs_cis = precompute_rope_freqs(args.dim // args.n_heads, args.max_seq_len)
        
        # 应用正态分布初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """工业级标准的大模型权重初始化"""
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
            # 去掉了冗余的 mask 参数传递
            h = layer(h, freqs_cis)
            
        h = self.norm(h)
        logits = self.output(h)
        
        loss = None
        if targets is not None:
            # 展平计算交叉熵损失 (自动忽略 -100)
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            
        return logits, loss