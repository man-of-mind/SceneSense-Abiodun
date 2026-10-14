"""Two report figures for the v20.4 association study (reads the evidence JSONs; GT already applied there)."""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
A = Path(sys.argv[1])
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
CUR, CAND = '#2a78d6', '#eb6834'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'text.color': INK, 'figure.facecolor': SURF, 'axes.facecolor': SURF, 'axes.titlesize': 10.5, 'axes.titleweight': 'semibold'})
# ---- figure 1: where split detections fall --------------------------------------------------------------------------
d = json.loads((A/'FRAGMENTATION_DIAGNOSIS.json').read_text())
rows = [r for c in d['clips'].values() for r in c['rows'] if r['cls'] == 'Pedestrian' and r['kind'] == 'SPLIT']
lat, rad = np.array([r['lateral'] for r in rows]), np.array([r['radial'] for r in rows])
inside = np.hypot(rad/2.7, lat/1.5) <= 1
fig, ax = plt.subplots(figsize=(7.2, 5.0))
t = np.linspace(0, np.pi/2, 200)
ax.plot(1.5*np.cos(t), 1.5*np.sin(t), color=INK2, lw=1.5, label='current gate: 1.5-m circle')
ax.plot(1.5*np.cos(t), 2.7*np.sin(t), color=INK2, lw=1.5, ls=(0, (4, 3)), label='candidate gate: ellipse, radial 2.7 m')
ax.scatter(lat, rad, s=34, color=CUR, edgecolor=SURF, linewidth=1.2, zorder=3, label=f'split detection (n={len(rows)})')
ax.set_xlabel('lateral offset from the live track (m)'); ax.set_ylabel('radial offset along the camera ray (m)')
ax.set_xlim(0, 3.2); ax.set_ylim(0, 6.2); ax.grid(color=GRID, lw=.8); ax.set_axisbelow(True)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
ax.set_title(f'Pedestrian track splits are range errors\n{int((rad > lat).sum())}/{len(rows)} radial-dominant; '
             f'{int(inside.sum())}/{len(rows)} fall inside the candidate ellipse', loc='left')
ax.legend(frameon=False, loc='upper right', labelcolor=INK)
fig.text(.01, .01, 'Accepted engine, 8 retained clips. Split = new display track for a person another live track already follows.\n'
         'GT labels detections for this diagnosis only.', fontsize=8, color=INK2)
fig.tight_layout(rect=(0, .06, 1, 1)); fig.savefig(A/'01-split-offsets-vs-gate.png', dpi=150); plt.close(fig)
# ---- figure 2: per-clip trade-off, current vs candidate -------------------------------------------------------------
e = json.loads((A/'ASSOCIATION_CANDIDATE_EVAL.json').read_text())
names = [('V20 ' if c['dir'] == 'v20' else 'V10 ')+c['clip'] for c in e['clips']]
def get(side, f):
    return [f(c[side]) for c in e['clips']]
panels = [('Duplicate pedestrian glyphs (fraction)  ↓', lambda s: s['identity']['Pedestrian']['duplicate_glyph_fraction']),
          ('Pedestrian ID switches (count)  ↓', lambda s: s['identity']['Pedestrian']['switches']),
          ('Track purity: share on one person  ↑', lambda s: s['identity']['Pedestrian']['track_purity']),
          ('Pedestrian step P99, m per frame  ↓', lambda s: s['identity']['Pedestrian']['step_m']['p99']),
          ('Moving-pedestrian error P50 (m)  ↓', lambda s: s['groups'].get('Pedestrian/moving', {}).get('error_m', {}).get('p50')),
          ('Walking person shown as moving  ↑', lambda s: s['pedestrian_moving_shown_as_moving'])]
fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), sharey=True)
y = np.arange(len(names))[::-1]
for ax, (title, f) in zip(axes.flat, panels):
    a, b = get('before', f), get('after', f)
    for yi, x0, x1 in zip(y, a, b):
        ax.plot([x0, x1], [yi, yi], color=GRID, lw=2, zorder=1)
    ax.scatter(a, y, s=48, color=CUR, edgecolor=SURF, linewidth=1.5, zorder=3, label='current (accepted)')
    ax.scatter(b, y, s=48, color=CAND, edgecolor=SURF, linewidth=1.5, zorder=3, label='candidate: ray gate')
    ax.set_title(title, loc='left'); ax.grid(axis='x', color=GRID, lw=.8); ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
    ax.tick_params(axis='y', length=0)
axes[0, 0].set_yticks(y, names); axes[1, 0].set_yticks(y, names)
h, l = axes[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc='upper left', bbox_to_anchor=(.005, .955), ncol=2, frameon=False, labelcolor=INK)
fig.suptitle('Ray-gated pedestrian association: fewer duplicates and switches, but more jumps and wrong-person merges', x=.01, ha='left', fontweight='semibold')
fig.text(.01, .01, '8 retained clips, 60 Hz lockstep, identical inputs. GT scores the display only. ↓ lower is better, ↑ higher is better. '
         'Vehicles are identical in both (0 differing rows).', fontsize=8, color=INK2)
fig.tight_layout(rect=(0, .03, 1, .92)); fig.savefig(A/'02-per-clip-current-vs-candidate.png', dpi=150); plt.close(fig)
print('ok', len(rows), int(inside.sum()))
