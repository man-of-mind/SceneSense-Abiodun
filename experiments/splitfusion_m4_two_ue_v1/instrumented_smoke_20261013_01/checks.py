"""Focused CPU checks for the M4 phase-marker/faulthandler instrumentation only."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile

OUT=Path(__file__).resolve().parent
ROOT=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else OUT.parents[2]
sys.path.insert(0,str(ROOT))
from rl_agent.splitfusion_m4_two_ue_v1 import phases, bootstrap


def rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines()]


def main():
    results={}
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp)
        log=phases.install(tmp)
        log.step('startup.a')
        written=rows(tmp/'PHASES.jsonl')  # visible before any close/flush call
        assert [r['event'] for r in written]==['EVENT','ENTER'] and written[1]['phase']=='startup.a'
        results['markers_unbuffered']=True
        log.step('startup.b')
        try:
            with log.phase('startup.inner'): raise KeyError('first')
        except KeyError as exc: log.fail_open_step(exc)
        try:
            with log.phase('teardown.x',stage='teardown'): raise OSError('cleanup')
        except OSError: pass
        first=json.loads((tmp/'FIRST_FAILURE.json').read_text())
        assert first['phase']=='startup.inner' and 'first' in first['error'] and first['traceback']
        events=[(r['event'],r['phase']) for r in rows(tmp/'PHASES.jsonl')]
        assert ('DONE','startup.a') in events and ('ERROR','startup.b') in events and ('ERROR','teardown.x') in events
        results['first_failure_separate_from_cleanup']=True
        # World teardown attempts every step, records each, keeps the original error.
        calls=[]
        def bad(): calls.append('bad'); raise RuntimeError('step failed')
        rows_=bootstrap.finish_world_resources([('one',bad),('two',lambda:calls.append('two'))],tmp,
                                               original_error=ValueError('orig'))
        assert calls==['bad','two'] and [r['status'] for r in rows_]==['FAILED','CLOSED']
        names=[(r['event'],r['phase'],r.get('stage')) for r in rows(tmp/'PHASES.jsonl')]
        assert ('ERROR','world_teardown.one','teardown') in names and ('DONE','world_teardown.two','teardown') in names
        assert json.loads((tmp/'FIRST_FAILURE.json').read_text())==first
        results['world_teardown_steps_marked']=True
        phases._ACTIVE=None
    # A native abort after ENTER: marker and faulthandler dump survive; returncode -6.
    with tempfile.TemporaryDirectory() as tmp:
        code=('import sys,os;sys.path.insert(0,%r);from rl_agent.splitfusion_m4_two_ue_v1 import phases;'
              'p=phases.install(%r);p.step("teardown.native",stage="teardown");os.abort()')%(str(ROOT),tmp)
        proc=subprocess.run([sys.executable,'-I','-c',code],capture_output=True,text=True)
        last=rows(Path(tmp)/'PHASES.jsonl')[-1]
        dump=(Path(tmp)/'FAULTHANDLER.log').read_text()
        assert proc.returncode==-6 and (last['event'],last['phase'])==('ENTER','teardown.native')
        assert 'Fatal Python error: Aborted' in dump and 'Current thread' in dump
        results['survives_native_abort']=dict(returncode=proc.returncode,last_marker=last['phase'],
                                              faulthandler_first_line=dump.splitlines()[0])
    tree=ast.parse((ROOT/'rl_agent/splitfusion_m4_two_ue_v1/host.py').read_text())
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    text=ast.unparse(run)
    assert text.index('phases.install(output)')<text.index('radio.start()')
    assert "P.phase('teardown.' + name, stage='teardown')" in text and 'P.fail_open_step(exc)' in text
    results['host_wiring']=True
    cli=(ROOT/'rl_agent/splitfusion_m4_two_ue_v1/cli.py').read_text()
    assert 'faulthandler.enable(all_threads=True)' in cli
    results['cli_faulthandler_stderr']=True
    with Path(sys.argv[1]).open('x') as f: json.dump(dict(status='PASS',root=str(ROOT),checks=results),f,indent=2)
    print('PASS',len(results))


if __name__=='__main__': main()
