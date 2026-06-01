import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 7.5,
    'axes.titlesize': 8.5,
    'axes.labelsize': 7.5,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.04,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'lines.linewidth': 1.2,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.4,
    'ytick.major.width': 0.4,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'xtick.major.pad': 2,
    'ytick.major.pad': 2,
})

# --- Data (rotation-fixed reeval, 30K) ---
lambdas =    [0,      0.005,  0.01,   0.1]
car_mean =   [54.75,  54.50,  56.33,  58.25]
car_std =    [3.18,   2.29,   1.61,   0.35]
n_seeds =    [2,      3,      3,      2]
labels =     ['0\n(LoRA-only)', '0.005', '0.01', '0.1']

car_mean = np.array(car_mean)
car_std = np.array(car_std)

C_LINE = '#1B4F7A'
C_FILL = '#A8D0F0'
C_BEST = '#E8734A'

fig, ax = plt.subplots(figsize=(3.2, 2.4))

x = np.arange(len(lambdas))

ax.fill_between(x, car_mean - car_std, car_mean + car_std,
                alpha=0.25, color=C_FILL, zorder=1)
ax.plot(x, car_mean, '-o', color=C_LINE, markersize=5, markeredgecolor='white',
        markeredgewidth=0.8, zorder=3)

# Highlight best point (lambda=0.1)
best_idx = np.argmax(car_mean)
ax.plot(x[best_idx], car_mean[best_idx], 'o', color=C_BEST, markersize=7,
        markeredgecolor='white', markeredgewidth=1.0, zorder=4)
ax.annotate(f'{car_mean[best_idx]:.2f}%',
            xy=(x[best_idx], car_mean[best_idx]),
            xytext=(x[best_idx] - 0.35, car_mean[best_idx] + 1.8),
            fontsize=6.5, fontweight='bold', color=C_BEST,
            arrowprops=dict(arrowstyle='->', color=C_BEST, lw=0.8,
                            shrinkA=0, shrinkB=3),
            zorder=5)

# Value labels for other points
for i in range(len(lambdas)):
    if i == best_idx:
        continue
    ax.text(x[i], car_mean[i] - car_std[i] - 1.2,
            f'{car_mean[i]:.2f}%', ha='center', va='top',
            fontsize=5.5, color='#555555')

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel(r'$\lambda$ (SDF loss weight)')
ax.set_ylabel('CAR (%)')
ax.set_ylim(48, 62)
ax.yaxis.grid(True, alpha=0.15, linewidth=0.4, zorder=0)

ax.set_title(r'Effect of $\lambda$ on Collision-Aware Rate',
             fontsize=7.5, fontstyle='italic', pad=6, loc='left')

os.makedirs('./figures', exist_ok=True)
fig.savefig('./figures/fig5_lambda_car.pdf')
fig.savefig('./figures/fig5_lambda_car.png')
plt.close()
print('Saved: figures/fig5_lambda_car.pdf + fig5_lambda_car.png')
