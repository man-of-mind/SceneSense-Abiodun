"""Refresh only the architecture handoff from the completed, sealed V11/V12 runs.

No project imports, inference, service startup, deployment or capture. Source
copies are mechanical and byte-identical; the presentation folder is not runnable.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

SELF = Path(__file__).resolve()
ROOT = SELF.parent.parent if SELF.parent.name == 'tools' else SELF.parent.parents[1]
OUT = ROOT / 'handoffs/supervisor_architecture_20261008_v1'
DEPLOY = ROOT.parent / 'abiodun_run4_single_gpu_profiler_v1'
HISTORY = OUT / 'history/20261008_v6'
ACTOR_SHA = 'bb9b0d256de79b4498cd622e1daa59edcf43ff3072dc9f675b265aac579ed3ce'
SPECS = (
    (11, 'GOOD_ALL', 'good_all_prospective_01', '25ce9744e96195162dfd4c4a995d5e71dbc9660bcbd2aae1c0826e0ebcbfe796'),
    (12, 'BAD_ALL', 'bad_all_prospective_02', '8cfbd1004f63effe4134bbf81e0692519da9610068f5fe6c2dd849e83adc3428'),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, document):
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + '\n')


def verify_copies(manifest):
    for row in manifest['files']:
        p = OUT / row['path']
        assert p.is_relative_to(OUT) and not p.is_symlink(), str(p)
        assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], str(p)


def selection():
    old = json.loads((OUT / 'SOURCE_MANIFEST.json').read_text())
    verify_copies(old)
    mapping = old['presentation_layout']['path_mapping']
    selected, registrations, evidence = {}, {}, {}

    def add(relative, expected, label, preferred=None):
        assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
        destination = mapping.get('code/' + relative, 'code/' + relative)
        choices = [preferred, DEPLOY / relative, ROOT / relative]
        source = next((p for p in choices if p and p.is_file() and sha(p) == expected), None)
        if source is None:
            raise FileNotFoundError('No registered source matches: ' + relative)
        previous = selected.get(destination)
        if previous:
            assert previous['sha256'] == expected, 'Conflicting runtime versions: ' + relative
            previous['reason'].append(label)
        else:
            selected[destination] = dict(path=destination, source=str(source), sha256=expected,
                                         bytes=source.stat().st_size, reason=[label])

    for version, condition, prospective, digest in SPECS:
        regpath = DEPLOY / f'rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261009_m3_route_b_v{version}' / prospective / 'registration.json'
        assert sha(regpath) == digest
        reg = json.loads(regpath.read_text())
        assert reg['channel']['schedule']['schedule_id'] == condition
        registrations[condition] = reg
        for relative, expected in reg['exact_gt_sources'].items():
            add(relative, expected, f'registered V{version} {condition} full Town10 map runtime')
        for row in reg['experiment_binds'].values():
            p = Path(row['path'])
            if p.suffix != '.py':
                continue
            base = next((base for base in (DEPLOY, ROOT) if p.is_relative_to(base)), None)
            assert base is not None, str(p)
            add(str(p.relative_to(base)), row['sha256'], f'registered V{version} launcher dependency', p)
        verdictpath = ROOT / f'experiments/splitfusion_m3_route_b_v{version}/registered_assessment_01/VERDICT_registered_binding.json'
        verdict = json.loads(verdictpath.read_text())
        assert verdict['verdict'] == 'PASS_M3_FULL_ROUTE_INTEGRATION'
        assert len(verdict['checks']) == 24 and all(verdict['checks'].values())
        evidence[condition] = dict(runtime_revision=reg['runtime_revision'], registration=str(regpath),
            registration_sha256=digest, authoritative_verdict=str(verdictpath),
            authoritative_verdict_sha256=sha(verdictpath), integration_status=verdict['verdict'],
            report=str(ROOT / f'experiments/splitfusion_m3_route_b_v{version}/LIVE_CAPTURE_REPORT.md'),
            note='References only; collected data, verdict documents and performance plots are not copied.')
    return old, selected, registrations, evidence


def apply_update():
    old, selected, registrations, evidence = selection()
    assert old['current_runtime'] == 'M3_1_ROUTE_B_GOOD_ALL_V6', 'Update already applied or different snapshot'
    assert not HISTORY.exists(), 'Preserve existing history; do not overwrite'
    HISTORY.mkdir(parents=True)
    for name in ('README.md', 'SOURCE_MANIFEST.json', 'RUNTIME_OVERVIEW.json', 'EXTERNAL_DEPENDENCIES.json'):
        shutil.copy2(OUT / name, HISTORY / name)
    (OUT / 'build_snapshot.py').rename(HISTORY / 'build_snapshot.py')
    (OUT / 'pending_fix').rename(HISTORY / 'pending_fix')
    records = {}
    for original in old['files']:
        row = dict(original)
        if row['path'] == 'build_snapshot.py' or row['path'].startswith('pending_fix/'):
            row['path'] = 'history/20261008_v6/' + row['path']
        records[row['path']] = row
    added, updated = [], []
    for destination, row in sorted(selected.items()):
        target = OUT / destination
        if destination not in records:
            added.append(destination)
        elif records[destination]['sha256'] != row['sha256']:
            backup = HISTORY / 'replaced_sources' / destination
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup)
            updated.append(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or sha(target) != row['sha256']:
            shutil.copyfile(row['source'], target)
        assert sha(target) == sha(row['source']) == row['sha256']
        combined = dict(records.get(destination, {}))
        combined.update(row)
        records[destination] = combined
    # Keep the user-friendly FCOS folders, original training figures and actor.
    assert sha(OUT / 'code/rl_agent/splitfusion_run5_full_live_v1/actor_export/actor_state_dict.pt') == ACTOR_SHA
    for row in old['files']:
        if row['path'].startswith(('code/fcos_model/', 'agent_training_plots/')):
            assert sha(OUT / row['path']) == row['sha256']
    radio = Path(evidence['GOOD_ALL']['registration']).parent / 'capture/m3_radio/radio_state_copy/effective_gnb_100mhz_4d5u_clean_minus50.conf'
    bad_radio = Path(evidence['BAD_ALL']['registration']).parent / 'capture/m3_radio/radio_state_copy/effective_gnb_100mhz_4d5u_clean_minus50.conf'
    assert sha(radio) == sha(bad_radio)
    radio_dest = 'radio_config/effective_gnb_100mhz_4d5u.conf'
    if sha(OUT / radio_dest) != sha(radio):
        shutil.copy2(OUT / radio_dest, HISTORY / 'effective_gnb_previous.conf')
        shutil.copyfile(radio, OUT / radio_dest)
    records[radio_dest] = dict(path=radio_dest, source=str(radio), sha256=sha(radio), bytes=radio.stat().st_size,
                             reason=['identical gNB configuration used by completed V11 and V12; configuration only'])
    refresh = ROOT / 'tools/refresh_supervisor_architecture_handoff.py'
    shutil.copyfile(refresh, OUT / 'refresh_snapshot.py')
    records['refresh_snapshot.py'] = dict(path='refresh_snapshot.py', source=str(refresh), sha256=sha(refresh),
                                        bytes=refresh.stat().st_size, reason=['bounded snapshot refresh and verification; never launches experiments'])
    overview_keys = ('runtime_revision', 'launcher_binding', 'policy', 'selection', 'route', 'channel',
                     'frozen_deadline_ms', 'map_class', 'numeric_prediction_contract', 'bookkeeping',
                     'radio_telemetry', 'gt_gc', 'pose_repair', 'capacity')
    for condition, filename in (('GOOD_ALL', 'RUNTIME_OVERVIEW.json'), ('BAD_ALL', 'RUNTIME_OVERVIEW_BAD_ALL.json')):
        overview = {key: registrations[condition][key] for key in overview_keys}
        overview['_handoff_note'] = 'Verbatim selected fields from this run registration. For current cross-condition status see README; inherited comments or fields describe that individual revision, not every later condition.'
        write_json(OUT / filename, overview)
    dependencies = json.loads((OUT / 'EXTERNAL_DEPENDENCIES.json').read_text())
    dependencies['carla'] = 'Native CARLA 0.10.0 installation and Python API are external. Current experiments launch CarlaUnreal.sh directly; they do not require the old carlasim/carla Docker image.'
    dependencies['runtime'] = 'Reading snapshot of the qualified V11/V12 system, not an authorized launch bundle. Original environment, sealed source inventories, registrations, OAI/core/edge images and external prerequisite evidence remain required.'
    dependencies['gt_accounting'] = 'Qualified V3 fast path runs only in the prep worker and requires checked CPython 3.10 capabilities; the worker refuses unsupported runtimes.'
    write_json(OUT / 'EXTERNAL_DEPENDENCIES.json', dependencies)
    manifest = dict(old)
    manifest.update(updated_utc=datetime.now(timezone.utc).isoformat(), current_runtime=registrations['BAD_ALL']['runtime_revision'],
        runtime_variants={key: value['runtime_revision'] for key, value in registrations.items()},
        reading_entrypoint='README.md', current_runtime_evidence=evidence,
        original_snapshot_preserved_at='history/20261008_v6',
        files=sorted(records.values(), key=lambda row: row['path']))
    write_json(OUT / 'SOURCE_MANIFEST.json', manifest)
    write_json(OUT / 'REFRESH_RECORD.json', dict(updated_utc=manifest['updated_utc'],
        purpose='Supervisor architecture handoff: working full Town10 map pipeline',
        newly_added_source_files=added, replaced_source_files=updated,
        registered_source_paths_covered=len(selected), copied_file_count=len(records),
        evidence_references=evidence, actor_sha256=ACTOR_SHA,
        fcos_sources_and_training_plots_unchanged=True, source_roots_not_modified=True,
        live_campaign_not_modified=True, no_services_or_inference_started=True,
        no_new_results_or_sensor_data_copied=True,
        historical_pending_fix='Moved under history/20261008_v6; active repair is now in code/ and registered in V11/V12.'))
    verify_copies(manifest)
    print(json.dumps(dict(added=len(added), replaced=len(updated), registered_sources=len(selected),
                          copied_files=len(records), output=str(OUT)), indent=2))


def verify(seal=False):
    manifest, selected, registrations, evidence = selection()
    assert manifest['current_runtime'] == registrations['BAD_ALL']['runtime_revision']
    rows = {row['path']: row for row in manifest['files']}
    for destination, row in selected.items():
        assert rows[destination]['sha256'] == row['sha256']
    python_count = 0
    for path in (OUT / 'code').rglob('*.py'):
        ast.parse(path.read_bytes(), filename=str(path))
        python_count += 1
    for target in re.findall(r'\]\(([^)]+)\)', (OUT / 'README.md').read_text()):
        if '://' not in target:
            assert (OUT / target.split('#', 1)[0]).exists(), target
    assert sha(OUT / 'code/rl_agent/splitfusion_run5_full_live_v1/actor_export/actor_state_dict.pt') == ACTOR_SHA
    hashes = {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*'))
              if p.is_file() and p.name != 'HANDOFF_SHA256.json'}
    if seal:
        write_json(OUT / 'HANDOFF_SHA256.json', hashes)
    else:
        assert hashes == json.loads((OUT / 'HANDOFF_SHA256.json').read_text())
    print(json.dumps(dict(status='PASS', file_hashes=len(hashes), python_syntax_checks=python_count,
                          registered_source_paths=len(selected), readme_links='PASS', no_runtime_imports=True), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan', action='store_true')
    mode.add_argument('--apply', action='store_true')
    mode.add_argument('--seal', action='store_true')
    mode.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.plan:
        old, selected, _, _ = selection()
        oldrows = {row['path']: row for row in old['files']}
        print(json.dumps(dict(registered_sources=len(selected),
            new=sum(path not in oldrows for path in selected),
            replacements=[path for path, row in selected.items() if path in oldrows and row['sha256'] != oldrows[path]['sha256']]), indent=2))
    elif args.apply:
        apply_update()
    else:
        verify(seal=args.seal)
