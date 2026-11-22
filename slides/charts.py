"""Builds the chart/illustration PNGs used by make_slides.py (scores = public leaderboard values from README)."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 15, 'axes.spines.top': False, 'axes.spines.right': False})
BLUE, PINK, GREEN, GREY = '#2F6FD6', '#E06C9F', '#2EA66B', '#9AA3B2'

def bars(fname, labels, vals, color, ylim, fmt, title):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    c = [GREY] * (len(vals) - 1) + [color]
    b = ax.bar(labels, vals, color=c, width=0.62)
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + (ylim[1] - ylim[0]) * 0.02, fmt.format(v), ha='center', fontsize=16, weight='bold')
    ax.set_ylim(*ylim); ax.set_title(title, fontsize=15, color='#555'); ax.tick_params(axis='x', labelsize=12.5)
    plt.tight_layout(); plt.savefig(f'img/{fname}', dpi=150); plt.close()

bars('t1_scores.png', ['Segmentation\nbaseline', '+ TrustMark\nverified rects', '+ PixelSeal\nhole filling', '+ rectangles\nre-aligned'],
     [0.6930, 0.7062, 0.7131, 0.7141], BLUE, (0.68, 0.72), '{:.4f}', 'Public mIoU (y-axis zoomed)')
bars('t3_scores.png', ['Delete the\n9 classes', 'Soften the\n9 classes', '+ reshape 36\nrelated classes', '+ tuned on\nholdout'],
     [0.0146, 0.0946, 0.1108, 0.1122], GREEN, (0, 0.13), '{:.4f}', 'Public score')

fig, ax = plt.subplots(figsize=(8.2, 4.6))
x = ['Text\ndistances', 'Persona\n+ T=1.0', '+ water-\nmarks', '+ sys.\nprompt', '+ ep-13\n(orig. C)', '+ ep-32\n(orig. B)']
y = [0.10, 0.45, 0.60, 0.70, 0.85, 1.00]
ax.plot(x, y, '-o', color=PINK, lw=3, ms=11)
for i, v in enumerate(y): ax.text(i, v + 0.05, f'{v:.2f}', ha='center', weight='bold', fontsize=16)
ax.set_ylim(0, 1.15); ax.set_title('Public score, one change per submission', fontsize=15, color='#555'); ax.tick_params(axis='x', labelsize=12)
plt.tight_layout(); plt.savefig('img/t2_scores.png', dpi=150); plt.close()

fig, axs = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
cls = ['true\nclass', 'B', 'C', 'D', 'E']
panels = [('Original model\n(saw the image)', [0.92, 0.03, 0.02, 0.02, 0.01], GREY),
          ('Naive: delete the class\n→ confidently wrong', [0.0, 0.90, 0.05, 0.03, 0.02], '#D9534F'),
          ('Ours ≈ retrained "gold" model\n→ uncertain', [0.38, 0.24, 0.16, 0.12, 0.10], GREEN)]
for ax, (t, p, c) in zip(axs, panels):
    ax.bar(cls, p, color=c); ax.set_title(t, fontsize=14); ax.set_ylim(0, 1); ax.tick_params(axis='x', labelsize=12)
axs[0].set_ylabel('probability')
fig.text(0.99, 0.01, 'illustration (not measured values)', ha='right', fontsize=10, color='#888')
plt.tight_layout(); plt.savefig('img/t3_idea.png', dpi=150); plt.close()
print('charts ok')
