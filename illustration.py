#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib as mpl

# mpl.rcParams.update(mpl.rcParamsDefault)

# for paper make the figure fonts larger
mpl.rcParams.update({
    'font.size': 12,            # Base font size
    'axes.titlesize': 14,       # Axes title
    'axes.labelsize': 17,       # X and Y axis labels
    'xtick.labelsize': 16,      # X tick labels
    'ytick.labelsize': 16,      # Y tick labels
    'legend.fontsize': 15,      # Legend font size
    'figure.titlesize': 14,     # Figure title (if used)
    # 'figure.dpi': 300,          # High-resolution figures
    'savefig.dpi': 300,         # Save figures at high resolution
    'pdf.fonttype': 42,         # Embed fonts in PDFs (useful for LaTeX compatibility)
    'ps.fonttype': 42
})

# Define x range
x = np.linspace(-5, 10, 1000)

# Define ground truth distribution
mu_gt, sigma_gt = 2, 1  # mean=2, std=1
ground_truth = norm(loc=mu_gt, scale=sigma_gt)

# Define predictions
# Optimal prediction (matches ground truth)
pred_optimal = norm(loc=mu_gt, scale=sigma_gt)

# Suboptimal prediction 1 (shifted mean)
pred_subopt1 = norm(loc=0, scale=1)

# Suboptimal prediction 2 (wider variance)
pred_subopt2 = norm(loc=2, scale=2)

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(x, ground_truth.pdf(x), label='Ground Truth', linewidth=2)
plt.plot(x, pred_optimal.pdf(x), '--', label='Prediction A (Optimal)', linewidth=2)
plt.plot(x, pred_subopt1.pdf(x), '--', label='Prediction B (Suboptimal 1)', linewidth=2)
plt.plot(x, pred_subopt2.pdf(x), '--', label='Prediction C (Suboptimal 2)', linewidth=2)

plt.title('Ground Truth and Prediction Densities')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(-4, 8)
plt.show()

# Data for the table
cell_text = [
    ["A", "1", "1"],
    ["B", "2", "3"],
    ["C", "3", "2"],
]
columns = ["Prediction", "Upstream Ranking", "Downstream Ranking"]

# Create figure and axis
fig, ax = plt.subplots(figsize=(6, 2))
ax.axis('off')  # Hide axes

# Add table
table = ax.table(
    cellText=cell_text,
    colLabels=columns,
    loc='center',
    cellLoc='center',
    edges='horizontal'  # Draw horizontal lines only
)

table.scale(1, 2)  # Scale the table (adjust height)
plt.title('Predictions and Their Ranking based on Performance ', pad=20)
plt.show()

#%%
np.random.seed(42)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define x range
x = np.linspace(-4, 8, 1000)

# Ground truth distribution
mu_gt, sigma_gt = 2, 1
ground_truth = norm(loc=mu_gt, scale=sigma_gt)

# Predictions
pred_optimal = norm(loc=mu_gt, scale=sigma_gt)
pred_subopt1 = norm(loc=0, scale=1)
pred_subopt2 = norm(loc=3, scale=2)

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [3, 2]})

# --- Left: Density plot ---
ax1.plot(x, ground_truth.pdf(x), label='Ground Truth', linewidth=2)
ax1.plot(x, pred_optimal.pdf(x), '--', label='Prediction (A)', linewidth=2)
ax1.plot(x, pred_subopt1.pdf(x), '--', label='Prediction (B)', linewidth=2)
ax1.plot(x, pred_subopt2.pdf(x), '--', label='Prediction (C)', linewidth=2)

ax1.set_title('Ground Truth and Predictions')
ax1.set_xlabel('x')
ax1.set_ylabel('Density')
ax1.legend()
ax1.grid(True)

# --- Right: Table ---
# Data for the table
cell_text = [
    ["A", "1", "1"],
    ["B", "2", "3"],
    ["C", "3", "2"],
]
columns = ["Prediction", "Upstream Rank", "Downstream Rank"]

# Hide axes
ax2.axis('off')

# Add table with bbox to control size
table = ax2.table(
    cellText=cell_text,
    colLabels=columns,
    loc='center',
    cellLoc='center',
    edges='horizontal',
    bbox=[0, 0, 1, 1]  # [left, bottom, width, height] in axes coordinates
)
table.auto_set_font_size(False)
table.set_fontsize(14)
table.scale(1, 2.5)  # Adjust scaling (height and width)

ax2.set_title('Predictions and Rankings', pad=20)

# Layout adjustment
plt.tight_layout()
plt.show()

#%%
# Define x range
x = np.linspace(-5, 10, 1000)

# Plotting
plt.figure(figsize=(6, 6))

plt.plot(x, ground_truth.pdf(x), label='Ground Truth', linewidth=3)
plt.plot(x, pred_optimal.pdf(x), '--', label='Prediction A', linewidth=3)
plt.plot(x, pred_subopt1.pdf(x), '--', label='Prediction B', linewidth=3)
plt.plot(x, pred_subopt2.pdf(x), '--', label='Prediction C', linewidth=3)

# plt.title('Ground Truth and Prediction Densities')
plt.xlabel('y')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(-4, 8)
plt.savefig('figs/illustration/predictions.pdf', dpi=300, bbox_inches='tight')
plt.show()
#%%

def f1(w_x):
    # return w_x**2
    return w_x**4
    # return (np.abs(w_x)**4) * (1.0 + 0.5 * np.sign(w_x))

def f2(x):
    return 0.5 *(x**4+2*x**3+2*x**2)
    # return 0.5 * (x**4 + 4*np.abs(x)**3)
    # return (np.abs(x)**4) * (1.0 + 0.95 * np.sign(x))

lo=-3
hi = 3
n_samples = 100
n_observations = 1000
# x = torch.linspace(lo, hi, 1000).unsqueeze(1)  # Input data
# x = torch.linspace(lo, hi, n_samples).repeat(n_observations,1)  # Input data
# x = np.linspace(lo, hi, num=n_samples).reshape(1,-1).repeat(1000, axis=0)
x = np.random.uniform(lo, hi, (n_observations,n_samples))
x = np.sort(x, axis=1) # sort w/r to the samples
# x = np.mean(x, axis=1)

f2_res = f2(x)
f2_min_i = np.argmin(f2_res, axis=1)
f1_res = f1(x)
f1_min_i = np.argmin(f1_res, axis=1)

def plot_funcs(x, f2_res, f2_min_i, f1_res, f1_min_i):
    plt.figure(figsize=(6, 6))
    plt.plot(x, f2_res, linewidth=3, label='Downstream Evaluation Function $f^d$')
    plt.plot(x[f2_min_i], f2_res[f2_min_i], '*', color='blue')
    plt.plot(x, f1(x), linewidth=3, label='Upstream Evaluation Function $f^u$')
    plt.plot(x[f1_min_i], f1_res[f1_min_i], '*', color='orange')
    plt.xlabel('z')
    plt.grid(alpha=0.25)
    plt.legend()
    plt.savefig('figs/illustration/functions.pdf', dpi=300, bbox_inches='tight')
    plt.show()

# plot for the ith instance and all scenarios
idx=0
plot_funcs(x[idx,:], f2_res[idx,:], f2_min_i[idx], f1_res[idx,:], f1_min_i[idx])

np.random.seed(42)

n_obs = 1000
n_samples = 500
obs    = ground_truth.rvs(size=(n_obs,1))
pred_a = pred_optimal.rvs(size=(n_obs,n_samples))
pred_b = pred_subopt1.rvs(size=(n_obs,n_samples))
pred_c = pred_subopt2.rvs(size=(n_obs,n_samples))

score_u_a = f1(obs-pred_a)
score_d_a = f2(obs-pred_a)

score_u_b = f1(obs-pred_b)
score_d_b = f2(obs-pred_b)

score_u_c = f1(obs-pred_c)
score_d_c = f2(obs-pred_c)

#%%
# Ideal scores
# plt.plot(score_d_a.mean(axis=1), score_d_a.mean(axis=1), color='k', label='Ideal', alpha=0.5)
# plt.plot(score_d_b.mean(axis=1), score_d_b.mean(axis=1), color='k', label='Ideal', alpha=0.5)
# plt.plot(score_d_c.mean(axis=1), score_d_c.mean(axis=1), color='k', label='Ideal', alpha=0.5)

xx = np.linspace(np.min(score_d_c), np.max(score_d_c), 100)
plt.figure(figsize=(6,6))
plt.plot(xx, xx, color='k', linestyle=':', label='Identity', alpha=0.9)
plt.hlines([], [], [], color='k', linestyle='--', label='Average score')
# scatter plot of scores 
plt.plot(score_u_a.mean(axis=1), score_d_a.mean(axis=1),'o', markersize=10, color='tab:orange', label='A', alpha=0.2)
plt.plot(score_u_b.mean(axis=1), score_d_b.mean(axis=1),'o', markersize=10, color='tab:green', label='B', alpha=0.2)
plt.plot(score_u_c.mean(axis=1), score_d_c.mean(axis=1),'o', markersize=10, color='tab:red',   label='C', alpha=0.3)
# upstream ranking (based on average score)
plt.vlines(score_u_a.mean(), 0, 200, linewidth=3, color='tab:orange', linestyle='--')
plt.vlines(score_u_b.mean(), 0, 200, linewidth=3, color='tab:green', linestyle='--')
plt.vlines(score_u_c.mean(), 0, 200, linewidth=3, color='tab:red', linestyle='--')
# plt.vlines([], [], [], color='k', linestyle='--', label='average score')
# downstream ranking (based on average score)
plt.hlines(score_d_a.mean(), 0, 200, linewidth=3, color='tab:orange', linestyle='--')
plt.hlines(score_d_b.mean(), 0, 200, linewidth=3, color='tab:green', linestyle='--')
plt.hlines(score_d_c.mean(), 0, 200, linewidth=3, color='tab:red', linestyle='--')
#
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18),
#           ncol=4, fancybox=True, shadow=True, labelspacing=0.5, handletextpad=0, columnspacing=0.5)

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0),
          ncol=4, fancybox=True, shadow=True, labelspacing=0.5, handletextpad=0.5, columnspacing=0.5)
#
# annotate ranks
scores_u = np.array([score_u_a.mean(), score_u_b.mean(), score_u_c.mean()])
scores_d = np.array([score_d_a.mean(), score_d_b.mean(), score_d_c.mean()])

# upstream rank: 1=best (lowest score) … etc.
ranks_u = np.argsort(np.argsort(scores_u)) + 1
ranks_d = np.argsort(np.argsort(scores_d)) + 1

lims = [min(scores_u.min(), scores_d.min()), max(scores_u.max(), scores_d.max())]

y0 = -2.2 # lims[0]  # bottom of plot, for upstream‐rank circles
x0 = 1.4 # lims[0]  # left   of plot, for downstream‐rank circles

names = ['A','B','C']
for i, name in enumerate(names):
    xu, yd = scores_u[i], scores_d[i]
    ru, rd = ranks_u[i], ranks_d[i]
    if name == 'A':
        facecolor = 'C1' 
    elif name == 'B':
        facecolor = 'C2'
    else:
        facecolor = 'C3'
    # upstream rank: centered on (xu, y0)
    plt.text(
        xu, y0,
        str(ru),
        ha='center', va='bottom',
        fontsize=14, color='white',
        bbox=dict(boxstyle='circle', facecolor=facecolor, alpha=0.9)
    )

    # downstream rank: centered on (x0, yd)
    plt.text(
        x0, yd,
        str(rd),
        ha='right', va='center',
        fontsize=14, color='white',
        bbox=dict(boxstyle='circle', facecolor=facecolor, alpha=0.9)
    )

#
plt.xlabel('Upstream Score')
plt.ylabel('Downstream Score')
plt.xlim([0,150])
plt.ylim([0,150])
plt.savefig('figs/illustration/ranking.pdf', dpi=300, bbox_inches='tight')
plt.show()

# %%
