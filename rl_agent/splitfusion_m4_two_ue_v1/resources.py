"""Only-owned-process lifecycle. No pkill, container sweep or evidence removal."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from .identity import require
from .configuration import write_new


def process_birth(pid):
    text = Path(f'/proc/{int(pid)}/stat').read_text()
    return text[text.rfind(')')+2:].split()[19]


class OwnedProcesses:
    def __init__(self, output):
        self.output = Path(output)
        self.children, self.logs, self.records = [], [], []

    def start(self, name, argv, *, cwd, env=None, cpus=None):
        require(all(r['name'] != name for r in self.records), 'duplicate owned process')
        command = (['taskset', '-c', ','.join(map(str,cpus))] if cpus else []) + list(map(str,argv))
        log = (self.output / (name + '.log')).open('xb')
        try:
            child = subprocess.Popen(command,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=log,
                                     stderr=log,start_new_session=True)
        except BaseException:
            log.close(); raise
        record = dict(name=name,pid=child.pid,pgid=os.getpgid(child.pid),birth=process_birth(child.pid),argv=command,group=True)
        require(record['pgid'] == child.pid, 'child lacks owned process group')
        self.children.append(child); self.logs.append(log); self.records.append(record)
        write_new(self.output / (name + '.ownership.json'),record)
        return child

    def adopt(self, name, child):
        """Record an explicitly returned child of a retained resource owner."""
        require(all(r['name']!=name for r in self.records),'duplicate adopted owner')
        record=dict(name=name,pid=child.pid,pgid=os.getpgid(child.pid),birth=process_birth(child.pid),
                    group=os.getpgid(child.pid)==child.pid,argv=list(child.args))
        self.children.append(child); self.records.append(record)
        write_new(self.output/(name+'.ownership.json'),record)

    def close(self):
        outcomes = []
        for child,record in reversed(list(zip(self.children,self.records))):
            row = dict(record)
            try:
                if child.poll() is None:
                    require(process_birth(child.pid) == record['birth'] and os.getpgid(child.pid) == record['pgid'],
                            'owned process identity changed; refuse signal')
                    for sig in (signal.SIGINT,signal.SIGTERM):
                        target='-'+str(record['pgid']) if record['group'] else str(record['pid'])
                        subprocess.run(['sudo','-n','kill','-'+str(int(sig)),'--',target],
                                       check=True,capture_output=True,timeout=10)
                        try:
                            child.wait(timeout=10); break
                        except subprocess.TimeoutExpired:
                            pass
                    require(child.poll() is not None, 'owned process did not stop; left for explicit recovery')
                row['returncode'] = child.returncode
            except BaseException as error:
                row['cleanup_error'] = repr(error)
            outcomes.append(row)
        for log in self.logs: log.close()
        return outcomes


def check_owned_state(directory):
    """Read-only recovery check. Does not signal a guessed/stale PID."""
    results=[]
    for path in sorted(Path(directory).glob('*.ownership.json')):
        row=json.loads(path.read_bytes())
        try:
            row['still_owned']=(process_birth(row['pid'])==row['birth'] and os.getpgid(row['pid'])==row['pgid'])
        except (FileNotFoundError,ProcessLookupError): row['still_owned']=False
        results.append(row)
    return results


def signal_owned_record(row):
    """An owner exiting during recovery is success, not an identity bypass."""
    def still_present():
        try:
            birth=process_birth(row['pid']); pgid=os.getpgid(row['pid'])
        except (FileNotFoundError,ProcessLookupError): return False
        require(birth==row['birth'] and pgid==row['pgid'],'PID/group changed during recovery; refuse signal')
        return True
    if not still_present(): return 'ALREADY_EXITED'
    target='-'+str(row['pgid']) if row.get('group',True) else str(row['pid'])
    try:
        subprocess.run(['sudo','-n','kill','-15','--',target],check=True,timeout=10,capture_output=True)
    except subprocess.CalledProcessError:
        if still_present(): raise
        return 'EXITED_DURING_SIGNAL'
    return 'SIGTERM_SENT'


def recover(directory, *, execute=False):
    """Explicit recovery; refuses active parent or unverifiable ownership.

    Normal teardown remains host.run's finally block. This recovery path never
    deletes outputs, and will not guess CARLA/radio ownership after PID reuse.
    """
    directory=Path(directory).resolve()
    from .configuration import ROOT
    require(directory!=ROOT and ROOT in directory.parents,'recovery must name one scoped capture directory')
    require((directory/'HOST_OWNER.json').is_file() and (directory/'PREPARATION_REFERENCE.json').is_file(),
            'capture ownership evidence missing')
    parent=json.loads((directory/'HOST_OWNER.json').read_bytes())
    try: alive=process_birth(parent['pid'])==parent['birth']
    except (FileNotFoundError,ProcessLookupError): alive=False
    records=check_owned_state(directory)
    result=dict(active_parent=alive,owned_processes=records,executed=False,
                normal_cleanup='host.run finally: both collectors -> edge -> both GT -> map -> radio restore -> owned processes')
    if not execute: return result
    require(not alive,'capture parent remains alive; let its finally block own teardown')
    # Restore only when the retained gNB owner can still be proved.
    if any(r['name']=='gnb' and r['still_owned'] for r in records):
        from rl_agent import ue_target_snr_cell_runtime_v1 as B
        oai=B.load_oai_module(); session=oai.TelnetSession('127.0.0.1',9090,2.,1<<20)
        try:
            *_,text=session.command('channelmod show current')
            models=oai.parse_channel_models(text)
            for name in ('rfsimu_channel_ue0','rfsimu_channel_ue1'):
                require(name in models,'recovery source channel absent')
                *_,response=session.command(f'channelmod modify {int(models[name]["model_index"])} noise_power_dB -50.0')
                B.validate_modify(response,'-50.0',oai)
            *_,text=session.command('channelmod show current')
            restored=oai.parse_channel_models(text)
            require(all(float(restored[n]['noise_power_db'])==-50. for n in ('rfsimu_channel_ue0','rfsimu_channel_ue1')),
                    'radio restoration readback failed')
        finally: session.close()
    container=directory/'EDGE_CONTAINER.json'
    if container.is_file():
        record=json.loads(container.read_bytes())
        probe=subprocess.run(['sudo','-n','docker','inspect',record['name']],capture_output=True,text=True,timeout=10)
        if probe.returncode==0:
            actual=json.loads(probe.stdout)[0]
            runtime=json.loads((directory/'RUNTIME.json').read_bytes())
            require(actual['Config']['Labels'].get('splitfusion.m4.owner')==runtime['sha256']
                    and actual['Image']==record['image'],'container ownership/image mismatch')
            subprocess.run(['sudo','-n','docker','stop','--time','30',actual['Id']],check=True,timeout=40,capture_output=True)
    for row in reversed(records):
        if row['still_owned']:
            row['recovery_signal']=signal_owned_record(row)
    result.update(executed=True,remaining=check_owned_state(directory))
    certificate=directory/'CARLA_STARTED.json'
    if certificate.is_file():
        from data_collection.run_route_b_collection_supervised_v1 import process_group_members
        record=json.loads(certificate.read_bytes())
        live=process_group_members(record['pgid'])
        known={p['pid']:p['birth'] for p in record['members']}
        if live:
            require(any(p['pid'] in known and process_birth(p['pid'])==known[p['pid']] for p in live),
                    'CARLA group no longer has a birth-certified owner; refuse recovery signal')
            os.killpg(record['pgid'],signal.SIGTERM)
            deadline=time.monotonic()+30
            while process_group_members(record['pgid']) and time.monotonic()<deadline: time.sleep(.2)
        result['carla_remaining']=process_group_members(record['pgid'])
    elif (directory/'CARLA_OWNER.json').is_file():
        result['carla_unverified']='Startup did not reach RPC certificate; inspect original startup ownership before recovery'
    return result
