"""Inventory and preserve worktree source changes before Git consolidation.

Read-only toward all worktrees. Writes recovery evidence below experiments/ only.
Raw captures, checkpoints, caches and .orig/.rej files are never selected for Git.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/git_consolidation_20261009'


def git(wt, *args):
    return subprocess.check_output(['git', '-C', str(wt), *args])


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def candidate(rel):
    p = Path(rel)
    if any(x in p.parts for x in ('experiments', 'campaign_runs', 'evaluation_runs', '__pycache__', '.pytest_cache')):
        return False
    if p.suffix in ('.orig', '.rej', '.pt', '.pth', '.log', '.raw', '.pcap', '.pcapng', '.gz', '.zip', '.tar'):
        return False
    return p.parts[0] in ('rl_agent', 'tools', 'handoffs')


def inventory():
    OUT.mkdir(exist_ok=True, parents=True)
    worktrees = [Path(s[9:]) for s in git(ROOT, 'worktree', 'list', '--porcelain').decode().splitlines()
                 if s.startswith('worktree ')]
    document = {'worktrees': [], 'refs': git(ROOT, 'show-ref').decode().splitlines(),
                'recovery_note': 'No raw experiment files are deleted, moved, or modified.'}
    members = []
    for wt in worktrees:
        short = 'main' if wt == ROOT else wt.name
        tracked = git(wt, 'diff', '--name-only', 'HEAD').decode().splitlines()
        untracked = [s for s in git(wt, 'ls-files', '--others', '--exclude-standard', '-z').decode().split('\0') if s]
        selected = [s for s in untracked if candidate(s) and (wt/s).is_file() and not (wt/s).is_symlink()]
        selected_rows = []
        for rel in sorted(selected):
            path = wt/rel
            s = path.stat().st_size
            if s > 10_000_000:
                raise RuntimeError(f'Unexpected large source candidate: {path}: {s}')
            entry = {'path': rel, 'bytes': s, 'sha256': digest(path)}
            if wt != ROOT and (ROOT/rel).is_file():
                entry['main_same'] = digest(ROOT/rel) == entry['sha256']
            selected_rows.append(entry)
            members.append((path, f'{short}/untracked/{rel}'))
        patch = git(wt, 'diff', '--binary', 'HEAD')
        patch_path = OUT / f'{short}.tracked.patch'
        patch_path.write_bytes(patch)
        tracked_rows = []
        for rel in tracked:
            path = wt/rel
            if path.is_file() and not path.is_symlink():
                tracked_rows.append({'path': rel, 'bytes': path.stat().st_size, 'sha256': digest(path)})
                members.append((path, f'{short}/tracked/{rel}'))
        extra = []
        for rel in untracked:
            if Path(rel).suffix in ('.orig', '.rej') and (wt/rel).is_file():
                members.append((wt/rel, f'{short}/merge_scratch/{rel}'))
                extra.append(rel)
        document['worktrees'].append({
            'path': str(wt), 'name': short, 'branch': git(wt, 'branch', '--show-current').decode().strip(),
            'head': git(wt, 'rev-parse', 'HEAD').decode().strip(),
            'tracked_changes': tracked_rows, 'source_candidates': selected_rows,
            'untracked_total': len(untracked), 'preserved_merge_scratch': extra,
            'patch_sha256': digest(patch_path)})
        oai = wt/'OAI/openairinterface5g'
        if (oai/'.git').exists():
            (OUT/f'{short}.oai.patch').write_bytes(git(oai, 'diff', '--binary', 'HEAD'))
    archive = OUT/'UNCOMMITTED_SOURCES.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for src, name in members:
            tar.add(src, arcname=name, recursive=False)
    with tarfile.open(archive, 'r:gz') as tar:
        for src, name in members:
            data = tar.extractfile(name).read()
            if hashlib.sha256(data).hexdigest() != digest(src):
                raise RuntimeError(f'Backup verification failed: {src}')
    document['source_archive'] = {'path': str(archive), 'sha256': digest(archive), 'members_verified': len(members)}
    (OUT/'INVENTORY.json').write_text(json.dumps(document, indent=2, sort_keys=True)+'\n')
    for wt in document['worktrees']:
        differences = [r['path'] for r in wt['source_candidates'] if r.get('main_same') is False]
        unique = [r['path'] for r in wt['source_candidates'] if 'main_same' not in r]
        print(json.dumps({'worktree':wt['name'], 'tracked':len(wt['tracked_changes']),
                          'candidates':len(wt['source_candidates']), 'bytes':sum(r['bytes'] for r in wt['source_candidates']),
                          'different_from_main':differences, 'unique':unique if wt['name']!='main' else len(unique)}))
    print('Verified recovery archive:', archive.stat().st_size, 'bytes;', len(members), 'files')


def prepare_checkpoint():
    inventory_doc = json.loads((OUT/'INVENTORY.json').read_text())
    main = next(w for w in inventory_doc['worktrees'] if w['name'] == 'main')
    paths = [r['path'] for r in main['source_candidates'] + main['tracked_changes']]
    paths += ['.gitignore', '.gitmodules']
    oai = ROOT/'OAI/openairinterface5g'
    patch_dir = ROOT/'OAI/local_changes'
    patch_dir.mkdir(exist_ok=True)
    patch_path = patch_dir/'telemetry_20261009.patch'
    patch = git(oai, 'diff', '--binary', 'HEAD')
    if patch != (OUT/'main.oai.patch').read_bytes():
        raise RuntimeError('OAI changed during consolidation')
    patch_path.write_bytes(patch)
    metadata = {
        'base_commit': git(oai, 'rev-parse', 'HEAD').decode().strip(),
        'origin': 'https://github.com/man-of-mind/openairinterface5g-scenesense.git',
        'patch': patch_path.name, 'sha256': digest(patch_path),
        'local_checkout_not_modified': True,
        'note': 'The gitlink retains the existing published base. These local telemetry/scheduler edits are preserved as a patch; no OAI commit or push was made.',
        'apply_on_clean_base_only': 'git apply --check ../local_changes/telemetry_20261009.patch && git apply ../local_changes/telemetry_20261009.patch',
    }
    meta_path = patch_dir/'telemetry_20261009.json'
    meta_path.write_text(json.dumps(metadata,indent=2,sort_keys=True)+'\n')
    paths += [str(p.relative_to(ROOT)) for p in (patch_path, meta_path)]
    for row in main['tracked_changes']:
        if digest(ROOT/row['path']) != row['sha256']:
            raise RuntimeError(f"Concurrent tracked change: {row['path']}")
    for row in main['source_candidates']:
        if row['path'] == 'tools/git_consolidation_20261009.py':
            continue
        if digest(ROOT/row['path']) != row['sha256']:
            raise RuntimeError(f"Concurrent source change: {row['path']}")
    paths = sorted(set(paths))
    (OUT/'MAIN_STAGE_PATHS.nul').write_bytes(b''.join(p.encode()+b'\0' for p in paths))
    (OUT/'MAIN_CHECKPOINT_FILES.json').write_text(json.dumps({p:digest(ROOT/p) for p in paths},indent=2,sort_keys=True)+'\n')
    print('Explicit checkpoint paths:',len(paths),'bytes:',sum((ROOT/p).stat().st_size for p in paths))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-checkpoint',action='store_true')
    args = parser.parse_args()
    prepare_checkpoint() if args.prepare_checkpoint else inventory()
