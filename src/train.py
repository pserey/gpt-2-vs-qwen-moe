import os, sys, time, math, argparse, yaml, json
import logging
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.gpt2_baseline import GPTModel
from models.qwen_moe import GPTModelMoE
from dataset import create_loaders

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("train")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(
        fmt='[%(levelname)s] %(asctime)s - %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    ))
    logger.addHandler(h)
level = os.getenv("LLM_LOG_LEVEL", "INFO").upper()
logger.setLevel(getattr(logging, level, logging.INFO))

# -----------------------------------------------------------------------------
def set_seed(s):
    import random, numpy as np
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def load_text(path):
    ok = os.path.exists(path)
    logger.info(f"Lendo arquivo: {path} (existe={ok})")
    return open(path, 'r', encoding='utf-8').read() if ok else ''

def cross_entropy_logits(logits, targets):
    return nn.functional.cross_entropy(logits.flatten(0,1), targets.flatten())

@torch.no_grad()
def calc_loss_loader(data_loader, model, device, eval_iter=5, is_moe=False, aux_weight=0.0, use_ckpt=False):
    """Avalia perda média em alguns batches (sem grad)."""
    total = 0.0; n = 0
    for i, (inp, tgt) in enumerate(data_loader):
        if i >= eval_iter: break
        inp, tgt = inp.to(device), tgt.to(device)
        if is_moe:
            logits, aux_loss = forward_model(model, inp, use_ckpt=use_ckpt)
            loss = cross_entropy_logits(logits, tgt) + (aux_weight * aux_loss if aux_loss > 0 else 0.0)
        else:
            logits = forward_model(model, inp, use_ckpt=use_ckpt)
            loss = cross_entropy_logits(logits, tgt)
        total += loss.item(); n += 1
    avg = total / max(1, n)
    logger.debug(f"calc_loss_loader -> iters={n}, loss={avg:.4f}")
    return avg

def generate_text(model, tokenizer_enc, prompt, max_new_tokens, context_size, is_moe=False, temperature=0.8, top_p=0.9, use_ckpt=False):
    """Amostra texto (nucleus). Usa o mesmo caminho de forward (com/sem checkpoint)."""
    logger.info(f"Geração qualitativa — prompt='{prompt[:60]}...' tokens={max_new_tokens}")
    model.eval()
    idx = torch.tensor(tokenizer_enc.encode(prompt)).unsqueeze(0).to(next(model.parameters()).device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        if is_moe:
            logits, _ = forward_model(model, idx_cond, use_ckpt=use_ckpt)  # Ignore aux_loss during generation
        else:
            logits = forward_model(model, idx_cond, use_ckpt=use_ckpt)
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumprobs = torch.cumsum(sorted_probs, dim=-1)
        keep = cumprobs <= top_p; keep[..., 0] = True
        sorted_probs = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        next_sorted = torch.multinomial(sorted_probs, num_samples=1)
        next_id = sorted_idx.gather(-1, next_sorted)
        idx = torch.cat([idx, next_id], dim=1)
    text = tokenizer_enc.decode(idx.squeeze(0).tolist())
    logger.debug("Geração concluída.")
    return text

def build_scheduler(optimizer, total_steps, warmup_steps, scheduler_name):
    if scheduler_name == 'cosine':
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        logger.info(f"Scheduler: cosine (steps={total_steps}, warmup={warmup_steps})")
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    logger.info("Scheduler: none")
    return None

# ---------- Checkpoint-aware forwards ----------
def _baseline_manual_forward(model: GPTModel, idx, use_ckpt: bool):
    B, T = idx.shape
    if model.pos_emb is None:
        x = model.tok_emb(idx)
    else:
        pos = torch.arange(T, device=idx.device)
        x = model.tok_emb(idx) + model.pos_emb(pos)[None, :, :]
    x = model.drop(x)
    if use_ckpt:
        x.requires_grad_(True)
    for bi, blk in enumerate(model.blocks):
        if use_ckpt:
            x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
        else:
            x = blk(x)
        logger.debug(f"[baseline] bloco {bi+1}/{len(model.blocks)} ok")
    x = model.ln(x)
    logits = model.head(x)
    return logits

def _moe_manual_forward(model: GPTModelMoE, idx, use_ckpt: bool):
    # For MoE, we don't use gradient checkpoint due to complexity with aux_loss
    # The model's forward already handles all the logic
    logits, aux_loss = model(idx)
    return logits, aux_loss

def forward_model(model, idx, use_ckpt: bool):
    if isinstance(model, GPTModelMoE):
        return _moe_manual_forward(model, idx, use_ckpt)
    else:
        return _baseline_manual_forward(model, idx, use_ckpt)

# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    logger.info(f"Lendo config: {args.config}")
    cfg = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))

    set_seed(cfg.get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Data
    train_txt = load_text(cfg['data']['train_path'])
    val_txt = load_text(cfg['data']['val_path'])
    test_txt = load_text(cfg['data']['test_path'])
    assert train_txt, 'Arquivo de treino não encontrado. Rode scripts/fetch_gutenberg_pt.py'
    from tiktoken import get_encoding
    tokenizer = get_encoding('gpt2')

    BS = cfg['training']['batch_size']
    CTX = cfg['training']['context_length']
    logger.info(f"DataLoader: batch_size={BS}, context_length={CTX}")
    train_loader, val_loader, test_loader = create_loaders(train_txt, val_txt, test_txt, batch_size=BS, max_length=CTX, stride=CTX)
    logger.info(f"Batches: train={len(train_loader)} val={len(val_loader)} test={len(test_loader)}")

    # Model
    model_cfg = {
        'vocab_size': cfg['model']['vocab_size'],
        'context_length': cfg['training']['context_length'],
        'emb_dim': cfg['model']['emb_dim'],
        'n_heads': cfg['model']['n_heads'],
        'n_layers': cfg['model']['n_layers'],
        'drop_rate': cfg['model']['drop_rate'],
        'qkv_bias': cfg['model'].get('qkv_bias', False),
        'rope': cfg['model'].get('rope', False),
        'mlp': cfg['model'].get('mlp', 'swiglu'),
    }
    
    # Check if this is a MoE model
    is_moe = cfg.get('model_type') == 'moe' and cfg['model'].get('num_experts', 0) > 0
    
    if is_moe:
        # Add MoE-specific parameters
        model_cfg.update({
            'num_experts': cfg['model']['num_experts'],
            'top_k': cfg['model'].get('top_k', 1),
            'capacity_factor': cfg['model'].get('capacity_factor', 1.0),
            'expert_hidden_dim': cfg['model'].get('expert_hidden_dim', 4 * cfg['model']['emb_dim']),
        })
        model = GPTModelMoE(model_cfg).to(device)
        logger.info(f"Modelo: Qwen-MoE — experts={model_cfg['num_experts']}, top_k={model_cfg['top_k']}")
    else:
        model = GPTModel(model_cfg).to(device)
        logger.info(f"Modelo: Baseline GPT-2")
        
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parâmetros treináveis totais: {n_params:,}")

    # Optimizer & scheduler
    optim = torch.optim.AdamW(model.parameters(), lr=cfg['training']['learning_rate'], weight_decay=cfg['training']['weight_decay'])
    total_steps = cfg['training']['num_epochs'] * len(train_loader) // max(1, cfg['training'].get('grad_accum_steps', 1))
    scheduler = build_scheduler(optim, total_steps, cfg['training'].get('warmup_steps', 0), cfg['training'].get('scheduler', None))
    scaler = GradScaler(enabled=cfg['training']['use_mixed_precision'])
    logger.info(f"Otimizador: AdamW lr={cfg['training']['learning_rate']} wd={cfg['training']['weight_decay']}")
    logger.info(f"Mixed precision: {cfg['training']['use_mixed_precision']} | Grad Accum: {cfg['training'].get('grad_accum_steps', 1)}")
    logger.info(f"Gradient checkpointing: {cfg['training'].get('gradient_checkpointing', False)}")

    # Engenharia
    grad_accum = max(1, cfg['training'].get('grad_accum_steps', 1))
    use_ckpt = cfg['training'].get('gradient_checkpointing', False)

    # Logging
    train_losses, val_losses, tokens_seen, throughput_hist = [], [], [], []
    total_tokens = 0; step = 0
    aux_weight = cfg['training'].get('aux_loss_weight', 0.01) if is_moe else 0.0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.time()
    optim.zero_grad(set_to_none=True)
    logger.info("===> Iniciando treinamento")
    for epoch in range(cfg['training']['num_epochs']):
        logger.info(f"--- Época {epoch+1}/{cfg['training']['num_epochs']} ---")
        model.train()
        for i, (inp, tgt) in enumerate(train_loader):
            inp, tgt = inp.to(device), tgt.to(device)
            with autocast(enabled=cfg['training']['use_mixed_precision']):
                if is_moe:
                    logits, aux_loss = forward_model(model, inp, use_ckpt=use_ckpt)
                    loss = cross_entropy_logits(logits, tgt) + (aux_weight * aux_loss if aux_loss > 0 else 0.0)
                else:
                    logits = forward_model(model, inp, use_ckpt=use_ckpt)
                    loss = cross_entropy_logits(logits, tgt)

            loss = loss / grad_accum
            scaler.scale(loss).backward()

            if (i + 1) % grad_accum == 0:
                if cfg['training']['grad_clip'] is not None:
                    scaler.unscale_(optim)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['grad_clip'])
                scaler.step(optim); scaler.update()
                optim.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                step += 1

                total_tokens += inp.numel() * grad_accum
                elapsed = time.time() - t0
                tok_interval = total_tokens - (tokens_seen[-1] if tokens_seen else 0)
                tps = tok_interval / max(1e-6, elapsed)
                throughput_hist.append(tps)
                t0 = time.time()

                if step % cfg['training']['eval_freq'] == 0:
                    tr = calc_loss_loader(train_loader, model, device, eval_iter=cfg['training']['eval_iter'], is_moe=is_moe, aux_weight=aux_weight, use_ckpt=use_ckpt)
                    va = calc_loss_loader(val_loader, model, device, eval_iter=cfg['training']['eval_iter'], is_moe=is_moe, aux_weight=aux_weight, use_ckpt=use_ckpt)
                    train_losses.append(tr); val_losses.append(va); tokens_seen.append(total_tokens)
                    logger.info(f"[eval] step={step} train_loss={tr:.3f} val_loss={va:.3f} TPS~{tps:.0f}")

    peak_mem = torch.cuda.max_memory_allocated(device) / (1024**3) if torch.cuda.is_available() else float('nan')

    # Test PPL
    model.eval()
    logger.info("===> Avaliando no conjunto de TESTE")
    with torch.no_grad():
        test_loss = calc_loss_loader(test_loader, model, device, eval_iter=50, is_moe=is_moe, aux_weight=aux_weight, use_ckpt=use_ckpt)
    ppl = math.exp(test_loss) if test_loss < 20 else float('inf')
    logger.info(f"Test PPL: {ppl:.2f} | Peak GPU (GB): {peak_mem:.2f}")

    # Save metrics
    save_dir = cfg.get('save_dir', 'results'); run_name = cfg.get('run_name', 'run')
    os.makedirs(os.path.join(save_dir, 'metrics'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'generations'), exist_ok=True)

    metrics_path = os.path.join(save_dir, 'metrics', f'{run_name}_metrics.json')
    metrics = {
        'model_type': cfg.get('model_type', 'baseline'),
        'run_name': run_name,
        'tokens_seen': tokens_seen,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'avg_tps': (sum(throughput_hist)/len(throughput_hist)) if throughput_hist else 0.0,
        'peak_mem_gb': peak_mem,
        'test_ppl': ppl,
        'gradient_checkpointing': use_ckpt,
    }
    
    # Add MoE-specific metrics
    if is_moe:
        metrics.update({
            'num_experts': model_cfg['num_experts'],
            'top_k': model_cfg['top_k'],
            'capacity_factor': model_cfg['capacity_factor'],
            'aux_loss_weight': aux_weight,
        })
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"Métricas salvas em: {metrics_path}")

    # Plot convergence
    if train_losses:
        import matplotlib.pyplot as plt
        xs = list(range(len(train_losses)))
        fig_path = os.path.join(save_dir, 'plots', f'{run_name}_convergence.png')
        plt.figure()
        plt.plot(xs, train_losses, label='Train')
        plt.plot(xs, val_losses, label='Val')
        plt.xlabel('Eval steps'); plt.ylabel('Loss'); plt.title(f'Convergência - {run_name}')
        plt.legend(); plt.tight_layout()
        plt.savefig(fig_path); plt.close()
        logger.info(f"Curva de convergência salva em: {fig_path}")

    # Qualitative generation - Portuguese Gutenberg style prompts
    prompts = [
        "O velho solar dormia entre montes, guardando um segredo que poucos ousavam recordar.",
        "Diziam os sábios que a razão humana era apenas um eco da memória divina.",
        "Nos tempos do Império, muitos temiam que a liberdade fosse mero nome de poder.",
        "No sertão do Ceará, o vento trazia poeira, lembranças e o canto dos vaqueiros.",
        "Segundo os economistas do reino, a prosperidade nascia mais da virtude que do ouro."
    ]
    try:
        import tiktoken
        enc = tiktoken.get_encoding('gpt2')
        gens = {}
        for i, p in enumerate(prompts, 1):
            text = generate_text(model, enc, p, max_new_tokens=120, context_size=cfg['training']['context_length'], is_moe=is_moe, use_ckpt=use_ckpt)
            gens[f'prompt_{i}'] = {'prompt': p, 'text': text}
        gen_path = os.path.join(save_dir, 'generations', f'{run_name}_generations.json')
        with open(gen_path, 'w', encoding='utf-8') as f:
            json.dump(gens, f, ensure_ascii=False, indent=2)
        logger.info(f"Gerações qualitativas salvas em: {gen_path}")
    except Exception as e:
        logger.exception('Geração qualitativa falhou')

if __name__ == '__main__':
    main()
