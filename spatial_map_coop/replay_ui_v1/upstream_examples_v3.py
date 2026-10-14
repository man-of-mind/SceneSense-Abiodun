"""OFFLINE illustration of the upstream limit: retained camera frame vs every raw
candidate (by score), archived GT and parked cars, and what v3 displays.
GT is used only for this figure; it never enters the UI.
Usage: python3 -m spatial_map_coop.replay_ui_v1.upstream_examples_v3 <v3_state.json> <gt_audit.json> <camera_dir> <out.png>
"""
import json, math, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from PIL import Image

STRONG = {'Vehicle': .20, 'Pedestrian': .25}


def to_ego(x, y, ego):
    a = math.radians(ego['yaw']); dx, dy = x - ego['x'], y - ego['y']
    # (right, forward) in the ego frame; CARLA yaw rotates +X (forward) toward +Y (right)
    return dy * math.cos(a) - dx * math.sin(a), dx * math.cos(a) + dy * math.sin(a)


def main(state_path, gt_path, cam_dir, out):
    state, gt = json.load(open(state_path)), json.load(open(gt_path))
    frames = {'1480': 'departure', '2020': 'turn', '2748': 'stop'}
    fig, axes = plt.subplots(3, 3, figsize=(19, 17), gridspec_kw=dict(width_ratios=[1.35, 1, 1]))
    fig.patch.set_facecolor('#0b1117')
    for row, (frame, clip) in enumerate(frames.items()):
        s, g = state[frame], gt['clips'][clip][frame]
        ego = s['ego'] or s['pose']
        ax = axes[row][0]; ax.imshow(Image.open(f'{cam_dir}/cam_{frame}.png')); ax.axis('off')
        ax.set_title(f'frame {frame} ({clip}) - retained model input RGB', color='w', fontsize=11)
        strong = sum(r['score'] >= STRONG[r['cls']] for r in s['raw'])
        for col, title in ((1, f'RAW: {len(s["raw"])} candidates ({strong} >= evaluator threshold)'),
                           (2, f'v3 DISPLAY: {sum(t["tier"]=="established" for t in s["tracks"])} tracked, '
                               f'{sum(t["tier"]=="tentative" for t in s["tracks"])} tentative, haze = unresolved')):
            ax = axes[row][col]; ax.set_facecolor('#0e161e'); ax.set_xlim(-30, 30); ax.set_ylim(-3, 47); ax.set_aspect('equal')
            ax.tick_params(colors='#8a9cab', labelsize=8); ax.set_title(title, color='w', fontsize=10)
            for sp in ax.spines.values(): sp.set_color('#2c3d4c')
            for px, py in gt['static_vehicles']:
                u, v = to_ego(px, py, ego); ax.add_patch(Rectangle((u - 1, v - 1), 2, 2, color='#3a4a58'))
            for a in g['actors']:
                u, v = to_ego(a[0], a[1], ego); ax.plot(u, v, 'o', ms=4, color='#5d6d7a')
            for p in g['positives']:
                u, v = to_ego(p['x'], p['y'], ego)
                ax.plot(u, v, 'o', ms=15, mfc='none', mec='#63e0bd', mew=2)
            ax.add_patch(Rectangle((-1, -2.4), 2, 4.8, color='#63e0bd'))
            if col == 1:
                for r in s['raw']:
                    u, v = to_ego(r['x'], r['y'], ego); st = r['score'] >= STRONG[r['cls']]
                    ax.plot(u, v, 'x' if st else '.', ms=9 if st else 5, color='#ffffff' if st else '#e7b45c', alpha=1 if st else .7)
            else:
                for c in s['field']:
                    u, v = to_ego(c['x'], c['y'], ego); ax.plot(u, v, 'o', ms=14, color='#e7b45c', alpha=min(.25, .05 + .06 * c['w']), mec='none')
                for t in s['tracks']:
                    u, v = to_ego(t['x'], t['y'], ego)
                    if t['cls'] == 'Pedestrian':
                        ax.plot(u, v, 'o', ms=8, color='#ffb07c', mfc='#ffb07c' if t['tier'] == 'established' else 'none'); continue
                    ang = (90 - (t['yaw'] - ego['yaw'])) if t['yaw'] is not None else 90
                    rect = Rectangle((-t['l'] / 2, -t['w'] / 2), t['l'], t['w'], fill=t['tier'] == 'established' and not t['conflict'],
                                     fc='#dce6ec', ec='#e7b45c' if t['conflict'] else '#dce6ec', ls='-' if t['tier'] == 'established' else '--')
                    rect.set_transform(Affine2D().rotate_deg(ang).translate(u, v) + ax.transData); ax.add_patch(rect)
    fig.text(.5, .005, 'Ego-forward metres. Green rings = archived GT visible actors (evaluation only). Grey dots = other dynamic actors; grey squares = parked static vehicles. '
             'White x = candidate at/above the registered evaluator threshold; amber dot = below it.', color='#b9c6d0', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .02, 1, 1)); fig.savefig(out, dpi=80, facecolor=fig.get_facecolor())


if __name__ == '__main__':
    main(*sys.argv[1:5])
