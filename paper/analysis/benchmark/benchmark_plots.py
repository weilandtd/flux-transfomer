"""
Version-comparison figures (old LabelingToFluxTransformer vs current FluxDecoder) on IO.
Reads the benchmark json files and emits guideline-style SVG+PNG figures.
"""
import json, pathlib, numpy as np, pandas as pd, torch
import matplotlib as mpl, matplotlib.pyplot as plt, seaborn as sns
mpl.rcParams.update({'font.family': 'Arial', 'font.size': 12})

COLORS = {'old': '#B0B0B0', 'current': '#4C72B0', 'ref': '#000000'}
D = pathlib.Path('paper/.results/version_benchmark')
CKPT = 'paper/.results/i_orientalis_20260620/imisor_transformer.pt'

old = json.load(open(D / 'old_model_io.json'))
old_map = {n: (np.nan if v is None else v) for n, v in zip(old['flux_names'], old['r2_per_reaction'])}

# current (as-designed) per-reaction R2 straight from the checkpoint's stored val correlations
ck = torch.load(CKPT, map_location='cpu', weights_only=False)
vc = ck['val_correlations']
cur_map = dict(zip(list(vc['flux_reaction_names']), np.asarray(vc['flux_r2_per_reaction'], float)))
cur_n_gt_09 = int(np.sum(np.asarray(vc['flux_r2_per_reaction'], float) > 0.9))

# ── Figure 1: per-reaction R2 scatter (old vs current, as-designed) ──────────
names = [n for n in old_map if n in cur_map]
o = np.array([old_map[n] for n in names]); c = np.array([cur_map[n] for n in names])
m = np.isfinite(o) & np.isfinite(c)
df = pd.DataFrame({'old': np.clip(o[m], -0.5, 1), 'current': np.clip(c[m], -0.5, 1)})
fig, ax = plt.subplots(figsize=(3.5, 3.5))
sns.scatterplot(data=df, x='old', y='current', s=14, color=COLORS['current'],
                edgecolor='none', alpha=0.5, ax=ax)
ax.plot([-0.5, 1], [-0.5, 1], '--', lw=0.8, color=COLORS['ref'])
ax.set_xlabel('Old model — per-reaction R²')
ax.set_ylabel('Current model — per-reaction R²')
ax.set_xlim(-0.5, 1.02); ax.set_ylim(-0.5, 1.02)
sns.despine(ax=ax)
fig.savefig(D / 'cmp_r2_scatter.svg', bbox_inches='tight')
fig.savefig(D / 'cmp_r2_scatter.png', dpi=300, bbox_inches='tight')

# ── Figure 2: # reactions R2>0.9 (as-designed) ───────────────────────────────
bar = pd.DataFrame({'version': ['Old', 'Current'],
                    'n': [old['n_r2_gt_0.9'], cur_n_gt_09]})
fig, ax = plt.subplots(figsize=(3.5, 3.5))
sns.barplot(data=bar, x='version', y='n', order=['Old', 'Current'],
            palette=[COLORS['old'], COLORS['current']], width=0.5, ax=ax)
for k, v in enumerate(bar['n']):
    ax.text(k, v + 4, str(int(v)), ha='center', va='bottom', fontsize=11)
ax.set_xlabel(''); ax.set_ylabel('Number of reactions (R² > 0.9)')
ax.set_ylim(0, max(bar['n']) * 1.18)
sns.despine(ax=ax)
fig.savefig(D / 'cmp_count_bar.svg', bbox_inches='tight')
fig.savefig(D / 'cmp_count_bar.png', dpi=300, bbox_inches='tight')

# ── Figure 3: old convergence curve (val loss) ───────────────────────────────
vc = old['val_curve']
dfc = pd.DataFrame({'epoch': np.arange(len(vc)), 'val_loss': vc})
fig, ax = plt.subplots(figsize=(3.5, 3.5))
sns.lineplot(data=dfc, x='epoch', y='val_loss', color=COLORS['old'], lw=1.2, ax=ax)
ax.axvline(144, color=COLORS['current'], ls='--', lw=0.8)   # current converged ~144
ax.set_yscale('log'); ax.set_xlabel('Epoch'); ax.set_ylabel('Old model — val loss')
sns.despine(ax=ax)
fig.savefig(D / 'cmp_old_convergence.svg', bbox_inches='tight')
fig.savefig(D / 'cmp_old_convergence.png', dpi=300, bbox_inches='tight')

print('saved figures to', D)
