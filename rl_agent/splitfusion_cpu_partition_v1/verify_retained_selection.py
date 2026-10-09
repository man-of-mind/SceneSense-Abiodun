"""Read-only validation of the accepted runtime selection; never starts services."""
import argparse
import hashlib
import json
from pathlib import Path

DEFAULT = Path(__file__).with_name('RETAINED_SINGLE_GPU_CONFIGURATION_V1.json')
MANIFEST_SHA = '498122d4399e3b10b0205f099fda0169ab5c69b458fd28102f9ac6e1dd32c2e5'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(selection=DEFAULT, repo=None):
    selection = Path(selection)
    config = json.loads(selection.read_text())
    if config['schema'] != 'splitfusion.retained_single_gpu_cpu_partition.v1':
        raise ValueError('Foreign selection schema')
    if config['evidence_manifest_sha256'] != MANIFEST_SHA or config['selected_arm'] != 'treatment':
        raise ValueError('Foreign qualification or non-treatment selection')
    repo = Path(repo) if repo is not None else Path(__file__).resolve().parents[2]
    evidence = (repo / config['evidence_root_relative_to_repo']).resolve()
    manifest_file = evidence / 'OUTPUT_SHA256.json'
    if sha(manifest_file) != MANIFEST_SHA:
        raise ValueError('Qualification manifest drift')
    records = json.loads(manifest_file.read_text())['files']
    index = {r['path']: r['sha256'] for r in records}

    def bound(name):
        path = evidence / name
        if sha(path) != index[name]:
            raise ValueError(f'Bound evidence drift: {name}')
        return json.loads(path.read_text())

    prospective = bound('PROSPECTIVE.json')
    result = bound('RESULT.json')
    for key in ('radio_cpus', 'edge_cpus', 'original_cpus', 'scope'):
        if config[key] != prospective[key]:
            raise ValueError(f'Selection differs from qualified setting: {key}')
    if set(config['radio_cpus']) & set(config['edge_cpus']):
        raise ValueError('Overlapping pools')
    if not result['comparison']['strict_prospective_acceptance']:
        raise ValueError('Qualification did not meet prospective acceptance')
    invariants = {'ue_omp_num_threads': 1, 'ue_mkl_num_threads': 1,
                  'radio_priority_change_allowed': False,
                  'apply_only_inside_owned_runtime_lifecycle': True,
                  'restore_before_teardown': True, 'persistent_host_affinity_change': False,
                  'cpu_radar': True, 'early_evaluator_submit': False,
                  'reward_deadline_ms': 170, 'quality_feedback_inside_deadline': True,
                  'automatic_launch': False, 'legacy_entry_points_automatically_replaced': False}
    for key, expected in invariants.items():
        if config[key] != expected:
            raise ValueError(f'Unqualified runtime setting: {key}')
    for name, expected in prospective['sources'].items():
        if sha(Path(name)) != expected:
            raise ValueError(f'Executed source drift: {name}')
    for arm in ('control', 'treatment'):
        qualified = bound(f'{arm}/capture/QUALIFICATION_RESULT.json')
        restored = bound(f'{arm}/RESTORATION_LATEST.json')
        if not qualified['passed'] or not restored['passed']:
            raise ValueError(f'Incomplete qualification/restoration: {arm}')
    return {'status': 'RETAINED_CONFIGURATION_VERIFIED', 'evidence_root': str(evidence),
            'selection_sha256': sha(selection), 'source_files_verified': len(prospective['sources']),
            'launch_performed': False, 'runtime_affinity_changed': False,
            'scope': config['scope'], 'selected_arm': 'treatment'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, default=DEFAULT)
    args = parser.parse_args()
    print(json.dumps(verify(args.selection), indent=2))
