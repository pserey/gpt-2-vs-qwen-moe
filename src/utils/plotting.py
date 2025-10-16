import json, matplotlib.pyplot as plt

def plot_convergence(metrics_json_path, out_png_path, title=None):
    with open(metrics_json_path, 'r', encoding='utf-8') as f:
        m = json.load(f)
    xs = list(range(len(m.get('train_losses', []))))
    plt.figure()
    plt.plot(xs, m.get('train_losses', []), label='Train')
    plt.plot(xs, m.get('val_losses', []), label='Val')
    plt.xlabel('Eval steps')
    plt.ylabel('Loss')
    plt.title(title or f"Convergência - {m.get('run_name','')}")
    plt.legend(); plt.tight_layout(); plt.savefig(out_png_path); plt.close()
