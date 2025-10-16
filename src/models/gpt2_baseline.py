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
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.Wq = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.Wk = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.Wv = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

        logger.debug(
            f"MultiHeadAttention init: d_in={d_in}, d_out={d_out}, heads={num_heads}, "
            f"head_dim={self.head_dim}, dropout={dropout}"
        )

    def forward(self, x):
        B, T, _ = x.shape
        logger.debug(f"MHA.forward in -> {tuple(x.shape)}")
        q = self.Wq(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.Wk(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.Wv(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        logger.debug(f"QKV shapes -> q:{tuple(q.shape)}, k:{tuple(k.shape)}, v:{tuple(v.shape)}")

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
        )
        self.ff = FeedForward(cfg['emb_dim'], cfg['drop_rate'], mlp=cfg.get('mlp','swiglu'))
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.drop = nn.Dropout(cfg['drop_rate'])

        logger.info(
            f"TransformerBlock init — emb_dim={cfg['emb_dim']} heads={cfg['n_heads']} "
            f"mlp={cfg.get('mlp','swiglu')} drop={cfg['drop_rate']}"
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
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop = nn.Dropout(cfg['drop_rate'])
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg['n_layers'])])
        self.ln = LayerNorm(cfg['emb_dim'])
        self.head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)
        self.cfg = cfg

        n_params = count_parameters(self)
        logger.info(
            f"GPTModel init — layers={cfg['n_layers']}, emb_dim={cfg['emb_dim']}, "
            f"heads={cfg['n_heads']}, mlp={cfg.get('mlp','swiglu')}, params_treináveis={n_params:,}"
        )

    def forward(self, idx):
        B, T = idx.shape
        logger.info(f"GPTModel.forward — batch={B}, seq_len={T}")
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)

        for bi, blk in enumerate(self.blocks):
            logger.debug(f"[forward] bloco {bi+1}/{len(self.blocks)} — início")
            x = blk(x)
            logger.debug(f"[forward] bloco {bi+1}/{len(self.blocks)} — fim")

        x = self.ln(x)
        out = self.head(x)
        logger.info(f"GPTModel.forward — saída logits {tuple(out.shape)}")
        return out
