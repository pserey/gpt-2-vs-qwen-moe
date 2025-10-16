import os
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
        y = self.w3(torch.nn.functional.silu(self.w1(x)) * self.w2(x))
        logger.debug(f"SwiGLU.forward -> {tuple(y.shape)}")
        return y

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
        self.ff = FeedForward(cfg['emb_dim'], cfg['drop_rate'], mlp=cfg.get('mlp','swiglu'))
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.drop = nn.Dropout(cfg['drop_rate'])

        logger.info(
            f"TransformerBlock init — emb_dim={cfg['emb_dim']} heads={cfg['n_heads']} "
            f"rope={cfg.get('rope', False)} mlp={cfg.get('mlp','swiglu')} drop={cfg['drop_rate']}"
        )

    def forward(self, x):
        logger.debug(f"Block.forward in -> {tuple(x.shape)}")
        s = x
        x = self.norm1(x); x = self.att(x); x = self.drop(x); x = x + s
        s = x
        x = self.norm2(x); x = self.ff(x); x = self.drop(x); x = x + s
        logger.debug(f"Block.forward out -> {tuple(x.shape)}")
        return x

class GPTModel(nn.Module):
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
        logger.info(
            f"GPTModel init — layers={cfg['n_layers']}, emb_dim={cfg['emb_dim']}, "
            f"heads={cfg['n_heads']}, rope={self.use_rope}, "
            f"mlp={cfg.get('mlp','swiglu')}, params_treináveis={n_params:,}"
        )

    def forward(self, idx):
        B, T = idx.shape
        logger.info(f"GPTModel.forward — batch={B}, seq_len={T}")
        
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

        for bi, blk in enumerate(self.blocks):
            logger.debug(f"[forward] bloco {bi+1}/{len(self.blocks)} — início")
            x = blk(x)
            logger.debug(f"[forward] bloco {bi+1}/{len(self.blocks)} — fim")

        x = self.ln(x)
        out = self.head(x)
        logger.info(f"GPTModel.forward — saída logits {tuple(out.shape)}")
        return out
