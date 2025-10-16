import os
import math
import logging
import torch
import torch.nn as nn

# --------------------------------------------------------------------
# Logging setup
# --------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        fmt='[%(levelname)s] %(asctime)s - %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    ))
    logger.addHandler(_h)

_env_level = os.getenv("LLM_LOG_LEVEL", "INFO").upper()
try:
    logger.setLevel(getattr(logging, _env_level))
except Exception:
    logger.setLevel(logging.INFO)
    logger.warning("LLM_LOG_LEVEL inválido; usando INFO.")

# --------------------------------------------------------------------
# Utils
# --------------------------------------------------------------------
def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

# --------------------------------------------------------------------
# RoPE (Rotary Position Embedding)
# --------------------------------------------------------------------
def apply_rope(x):
    """
    Apply Rotary Position Embedding to query/key tensors.
    
    Args:
        x: tensor of shape (B, num_heads, T, head_dim)
    
    Returns:
        x_rotated: tensor with RoPE applied
    """
    B, H, T, Dh = x.shape
    device = x.device
    
    # RoPE parameters
    theta = 10000.0
    
    # Position indices
    pos = torch.arange(T, device=device, dtype=torch.float)
    
    # Frequency computation
    inv_freq = torch.pow(theta, -2 * torch.arange(0, Dh, 2, device=device, dtype=torch.float) / Dh)
    
    # Create frequency matrix
    freqs = torch.einsum('t,f->tf', pos, inv_freq)  # (T, Dh//2)
    
    # Create cos and sin matrices
    cos = torch.cos(freqs).unsqueeze(0).unsqueeze(0)  # (1, 1, T, Dh//2)
    sin = torch.sin(freqs).unsqueeze(0).unsqueeze(0)  # (1, 1, T, Dh//2)
    
    # Split x into even and odd dimensions
    x1 = x[..., 0::2]  # Even dimensions (B, H, T, Dh//2)
    x2 = x[..., 1::2]  # Odd dimensions (B, H, T, Dh//2)
    
    # Apply rotation
    x_rotated = torch.zeros_like(x)
    x_rotated[..., 0::2] = x1 * cos - x2 * sin
    x_rotated[..., 1::2] = x1 * sin + x2 * cos
    
    return x_rotated

# --------------------------------------------------------------------
# Core layers
# --------------------------------------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(d))
        self.shift = nn.Parameter(torch.zeros(d))
    def forward(self, x):
        m = x.mean(dim=-1, keepdim=True)
        v = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.scale * (x - m) / torch.sqrt(v + self.eps) + self.shift

class GELU(nn.Module):
    def forward(self, x):
        return torch.nn.functional.gelu(x)

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=True)
        self.w2 = nn.Linear(dim, hidden_dim, bias=True)
        self.w3 = nn.Linear(hidden_dim, dim, bias=True)
    def forward(self, x):
        return self.w3(torch.nn.functional.silu(self.w1(x)) * self.w2(x))

# --------------------------------------------------------------------
# MoE Components
# --------------------------------------------------------------------
class Expert(nn.Module):
    """Individual expert using SwiGLU activation"""
    def __init__(self, emb_dim, hidden_dim):
        super().__init__()
        self.swiglu = SwiGLU(emb_dim, hidden_dim)
        
    def forward(self, x):
        return self.swiglu(x)

class Router(nn.Module):
    """Top-k router for selecting experts"""
    def __init__(self, emb_dim, num_experts, top_k=1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(emb_dim, num_experts, bias=False)
        
    def forward(self, x):
        # x: (batch_size, seq_len, emb_dim)
        scores = self.gate(x)  # (batch_size, seq_len, num_experts)
        
        # Get top-k experts and their scores
        topk_scores, topk_indices = torch.topk(scores, self.top_k, dim=-1)
        
        # Apply softmax to top-k scores for normalization
        topk_probs = torch.softmax(topk_scores, dim=-1)
        
        return topk_probs, topk_indices, scores

def compute_aux_loss(scores, topk_indices, num_experts):
    """
    Compute auxiliary loss for load balancing (Switch Transformer style)
    
    Args:
        scores: router scores (batch_size, seq_len, num_experts)
        topk_indices: selected expert indices (batch_size, seq_len, top_k)
        num_experts: total number of experts
    
    Returns:
        aux_loss: balancing loss to encourage uniform expert usage
    """
    batch_size, seq_len, _ = scores.shape
    
    # Compute gate probabilities (softmax over all experts)
    gate_probs = torch.softmax(scores, dim=-1)  # (batch_size, seq_len, num_experts)
    
    # Compute fraction of tokens assigned to each expert
    expert_counts = torch.zeros(num_experts, device=scores.device)
    for expert_id in range(num_experts):
        # Count how many tokens were assigned to this expert
        expert_mask = (topk_indices == expert_id).any(dim=-1)  # (batch_size, seq_len)
        expert_counts[expert_id] = expert_mask.float().sum()
    
    # Normalize to get fractions
    total_tokens = batch_size * seq_len
    expert_fractions = expert_counts / total_tokens  # (num_experts,)
    
    # Compute average gate probability for each expert
    avg_gate_probs = gate_probs.mean(dim=(0, 1))  # (num_experts,)
    
    # Aux loss = num_experts * sum(expert_fraction * avg_gate_prob)
    aux_loss = num_experts * torch.sum(expert_fractions * avg_gate_probs)
    
    return aux_loss

class MoEFeedForward(nn.Module):
    """
    Qwen-style Mixture of Experts Feed Forward layer
    
    Features:
    - Top-k expert selection (k=1 or 2)
    - Capacity factor for load balancing
    - Auxiliary loss for uniform expert usage
    - SwiGLU activation in experts
    """
    def __init__(self, cfg):
        super().__init__()
        self.emb_dim = cfg['emb_dim']
        self.num_experts = cfg['num_experts']
        self.top_k = cfg.get('top_k', 1)
        self.capacity_factor = cfg.get('capacity_factor', 1.0)
        self.hidden_dim = cfg.get('expert_hidden_dim', 4 * cfg['emb_dim'])
        
        # Router for expert selection
        self.router = Router(self.emb_dim, self.num_experts, self.top_k)
        
        # Create experts
        self.experts = nn.ModuleList([
            Expert(self.emb_dim, self.hidden_dim) 
            for _ in range(self.num_experts)
        ])
        
        logger.info(
            f"MoE init — experts={self.num_experts}, top_k={self.top_k}, "
            f"capacity_factor={self.capacity_factor}, expert_hidden_dim={self.hidden_dim}"
        )
        
    def forward(self, x):
        batch_size, seq_len, emb_dim = x.shape
        original_shape = x.shape
        
        # Flatten for easier processing
        x_flat = x.reshape(-1, emb_dim)  # (batch_size * seq_len, emb_dim)
        
        # Route tokens to experts
        topk_probs, topk_indices, all_scores = self.router(x)
        
        # Flatten routing outputs
        topk_probs_flat = topk_probs.reshape(-1, self.top_k)  # (batch_size * seq_len, top_k)
        topk_indices_flat = topk_indices.reshape(-1, self.top_k)  # (batch_size * seq_len, top_k)
        
        # Initialize output
        output_flat = torch.zeros_like(x_flat)
        
        # Process each expert
        unique_experts = torch.unique(topk_indices_flat)
        
        for expert_id_tensor in unique_experts:
            expert_id = int(expert_id_tensor.item())
            
            # Find tokens assigned to this expert
            expert_mask = (topk_indices_flat == expert_id)
            if not expert_mask.any():
                continue
                
            # Get tokens assigned to this expert (considering top-k)
            token_mask = expert_mask.any(dim=-1)
            selected_tokens_idx = token_mask.nonzero(as_tuple=False).squeeze(-1)
            
            if selected_tokens_idx.numel() == 0:
                continue
                
            # Apply capacity factor (limit tokens per expert)
            capacity = int(self.capacity_factor * len(x_flat) / self.num_experts)
            if len(selected_tokens_idx) > capacity:
                # Randomly sample tokens within capacity
                perm = torch.randperm(len(selected_tokens_idx), device=x.device)
                selected_tokens_idx = selected_tokens_idx[perm[:capacity]]
                logger.debug(f"Expert {expert_id}: capacity limit applied ({capacity} tokens)")
            
            # Get input for this expert
            expert_input = x_flat[selected_tokens_idx]  # (num_tokens, emb_dim)
            
            # Process through expert
            expert_output = self.experts[expert_id](expert_input)
            
            # Get corresponding probabilities for selected tokens
            expert_mask_selected = expert_mask[selected_tokens_idx]  # (num_tokens, top_k)
            
            # Find which position in top_k corresponds to this expert
            expert_positions = expert_mask_selected.int().argmax(dim=-1, keepdim=True)
            expert_probs = torch.gather(
                topk_probs_flat[selected_tokens_idx], 
                dim=-1, 
                index=expert_positions
            ).squeeze(-1)  # (num_tokens,)
            
            # Weight expert output by routing probability
            weighted_output = expert_output * expert_probs.unsqueeze(-1)
            
            # Add to final output
            output_flat.index_add_(0, selected_tokens_idx, weighted_output)
        
        # Compute auxiliary loss for load balancing
        aux_loss = compute_aux_loss(all_scores, topk_indices, self.num_experts)
        
        # Reshape back to original shape
        output = output_flat.reshape(original_shape)
        
        return output, aux_loss

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False, rope=False):
        super().__init__()
        assert d_out % num_heads == 0
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.rope = rope
        self.Wq = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.Wk = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.Wv = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

        logger.debug(
            f"MultiHeadAttention init: d_in={d_in}, d_out={d_out}, heads={num_heads}, "
            f"head_dim={self.head_dim}, rope={self.rope}, dropout={dropout}"
        )

    def forward(self, x):
        B, T, _ = x.shape
        logger.debug(f"MHA.forward in -> {tuple(x.shape)}")
        q = self.Wq(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.Wk(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.Wv(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        logger.debug(f"QKV shapes -> q:{tuple(q.shape)}, k:{tuple(k.shape)}, v:{tuple(v.shape)}")

        # Apply RoPE if enabled
        if self.rope:
            q = apply_rope(q)
            k = apply_rope(k)
            logger.debug("RoPE aplicado a Q e K")

        att = q @ k.transpose(2, 3) / (self.head_dim ** 0.5)
        mask_bool = self.mask.bool()[:T, :T]
        att.masked_fill_(mask_bool, float('-inf'))
        w = torch.softmax(att, dim=-1)
        w = self.dropout(w)
        ctx = (w @ v).transpose(1, 2).reshape(B, T, self.d_out)
        out = self.out_proj(ctx)
        logger.debug(f"MHA.forward out -> {tuple(out.shape)}")
        return out

class FeedForward(nn.Module):
    def __init__(self, emb_dim, drop_rate, mlp='swiglu'):
        super().__init__()
        if mlp == 'gelu':
            self.net = nn.Sequential(
                nn.Linear(emb_dim, 4 * emb_dim),
                GELU(),
                nn.Linear(4 * emb_dim, emb_dim),
                nn.Dropout(drop_rate),
            )
            logger.debug("FeedForward init: GELU")
        else:
            self.net = nn.Sequential(
                SwiGLU(emb_dim, 4 * emb_dim),
                nn.Dropout(drop_rate),
            )
            logger.debug("FeedForward init: SwiGLU")
    def forward(self, x):
        y = self.net(x)
        logger.debug(f"FFN.forward -> {tuple(y.shape)}")
        return y

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.att = MultiHeadAttention(
            d_in=cfg['emb_dim'], d_out=cfg['emb_dim'],
            context_length=cfg['context_length'], dropout=cfg['drop_rate'],
            num_heads=cfg['n_heads'], qkv_bias=cfg.get('qkv_bias', False),
            rope=cfg.get('rope', False),
        )
        
        # Use MoE if specified, otherwise regular FFN
        if cfg.get('num_experts', 0) > 0:
            self.ff = MoEFeedForward(cfg)
            self.use_moe = True
        else:
            self.ff = FeedForward(cfg['emb_dim'], cfg['drop_rate'], mlp=cfg.get('mlp','swiglu'))
            self.use_moe = False
            
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.drop = nn.Dropout(cfg['drop_rate'])

        logger.info(
            f"TransformerBlock init — emb_dim={cfg['emb_dim']} heads={cfg['n_heads']} "
            f"rope={cfg.get('rope', False)} mlp={cfg.get('mlp','swiglu')} "
            f"moe={'yes' if self.use_moe else 'no'} drop={cfg['drop_rate']}"
        )

    def forward(self, x):
        logger.debug(f"Block.forward in -> {tuple(x.shape)}")
        
        # Attention block with residual connection
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop(x)
        x = x + shortcut
        
        # Feed-forward block with residual connection
        shortcut = x
        x = self.norm2(x)
        
        if self.use_moe:
            x, aux_loss = self.ff(x)
        else:
            x = self.ff(x)
            aux_loss = 0.0
            
        x = self.drop(x)
        x = x + shortcut
        
        logger.debug(f"Block.forward out -> {tuple(x.shape)}")
        return x, aux_loss

class GPTModelMoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.use_rope = cfg.get('rope', False)
        # Only create positional embeddings if not using RoPE
        self.pos_emb = None if self.use_rope else nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop = nn.Dropout(cfg['drop_rate'])
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg['n_layers'])])
        self.ln = LayerNorm(cfg['emb_dim'])
        self.head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)
        self.cfg = cfg

        n_params = count_parameters(self)
        n_experts = cfg.get('num_experts', 0)
        active_params = self._estimate_active_params(cfg)
        
        logger.info(
            f"GPTModelMoE init — layers={cfg['n_layers']}, emb_dim={cfg['emb_dim']}, "
            f"heads={cfg['n_heads']}, rope={self.use_rope}, "
            f"experts={n_experts}, top_k={cfg.get('top_k', 1)}, "
            f"mlp={cfg.get('mlp','swiglu')}, params_total={n_params:,}, "
            f"params_ativos_por_step~{active_params:,}"
        )
        
    def _estimate_active_params(self, cfg):
        """Estimate active parameters per forward pass for MoE model"""
        n_experts = cfg.get('num_experts', 0)
        if n_experts == 0:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # For MoE: non-expert params + (top_k / num_experts) * expert_params
        top_k = cfg.get('top_k', 1)
        expert_ratio = top_k / n_experts
        
        # Rough estimation (actual calculation would be more complex)
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Estimate that ~20% of params are in experts (rough approximation)
        expert_params = int(0.2 * total_params)
        non_expert_params = total_params - expert_params
        
        active_params = non_expert_params + int(expert_ratio * expert_params)
        return active_params

    def forward(self, idx):
        B, T = idx.shape
        logger.info(f"GPTModelMoE.forward — batch={B}, seq_len={T}")
        
        # Token embeddings
        x = self.tok_emb(idx)
        
        # Add positional embeddings only if not using RoPE
        if not self.use_rope:
            pos = torch.arange(T, device=idx.device)
            x = x + self.pos_emb(pos)[None, :, :]
            logger.debug("Positional embeddings aplicados")
        else:
            logger.debug("Usando RoPE - sem positional embeddings")
        
        x = self.drop(x)

        # Accumulate auxiliary losses from MoE layers
        total_aux_loss = 0.0
        
        for bi, blk in enumerate(self.blocks):
            logger.debug(f"[forward] bloco {bi+1}/{len(self.blocks)} — início")
            x, aux_loss = blk(x)
            total_aux_loss += aux_loss
            logger.debug(f"[forward] bloco {bi+1}/{len(self.blocks)} — fim, aux_loss={aux_loss:.4f}")

        x = self.ln(x)
        out = self.head(x)
        
        logger.info(f"GPTModelMoE.forward — saída logits {tuple(out.shape)}, total_aux_loss={total_aux_loss:.4f}")
        return out, total_aux_loss
