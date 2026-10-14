"""Prospective two-logical-UE radio composition. Never called by offline checks."""
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import socket
import threading
import time
import uuid

from .identity import require
from .configuration import ROOT, CPU, sha, write_new
from .radio import EXPECTED, resolve, discover_interfaces
from rl_agent.multiue_oai.runner import parse_uicc_profiles


def uicc_block(text, number):
    match=re.search(r'(?m)^\s*uicc'+str(number)+r'\s*=\s*\{',text)
    require(match is not None,'missing registered UICC block')
    depth=1; end=match.end()
    while end<len(text) and depth:
        depth += (text[end]=='{')-(text[end]=='}'); end+=1
    require(depth==0,'unbalanced UICC configuration')
    if text[end:end+1]==';': end+=1
    return text[match.start():end]


def validate_ue_pair(text):
    """Validate the explicit lists, not CLI defaults OAI ignores for cells.

    Deliberately accepts only the known template's simple list/group shape.
    Native libconfig parsing of the generated file is also qualified offline.
    """
    clean=re.sub(r'#[^\n]*','',text)
    def groups(name):
        matches=list(re.finditer(r'(?ms)^'+name+r'\s*=\s*\((.*?)^\);',clean))
        require(len(matches)==1,'missing/duplicate UE '+name+' list')
        body=matches[0].group(1)
        rows=re.findall(r'\{([^{}]*)\}',body,re.S)
        require(len(rows)==2 and not re.sub(r'\{[^{}]*\}|[\s,]','',body,flags=re.S),
                'exactly two simple UE '+name+' entries required')
        result=[]
        for row in rows:
            pairs=re.findall(r'(\w+)\s*=\s*([^;]+);',row)
            require(len(pairs)==len(dict(pairs)),'duplicate UE '+name+' field')
            require(not re.sub(r'\w+\s*=\s*[^;]+;|\s','',row),'invalid UE '+name+' field')
            result.append({k:re.sub(r'\s','',v) for k,v in pairs})
        return result
    cells=groups('cells'); rus=groups('RUs'); simulators=groups('rfsimulator')
    for i in range(2):
        require(cells[i]==dict(ru_id=str(i),band='78',rf_freq='3649260000L',numerology='1',
                              N_RB_DL='273',ssb_start='516'),'UE cell/carrier mapping differs: '+str(i))
        require(rus[i]==dict(nb_tx='1',nb_rx='1'),'UE radio-unit/antenna mapping differs')
        require(simulators[i]==dict(serveraddr='"127.0.0.1"',options='("chanmod")'),
                'UE simulator/channel modelling differs')
    profiles=parse_uicc_profiles(text)
    require(set(profiles)=={0,1} and all(profiles[i]['imsi']==v['imsi'] for i,v in enumerate(EXPECTED)),
            'prospective UICC/source relationships differ')
    return dict(cells=cells,radio_units=rus,simulators=simulators,ue_count=2)


def prepare_radio(output):
    """Qualified 273-PRB/4D5U carrier + full two-UE template, no live writes."""
    from rl_agent import splitfusion_phase14a_100mhz_calibration_v1 as P
    config=json.loads((ROOT/'rl_agent/configs/splitfusion_phase14a_100mhz_calibration_v1.json').read_bytes())
    for key,digest in config['source_sha256'].items():
        if key in config['paths']:
            require(sha(ROOT/config['paths'][key])==digest,'radio source drift: '+key)
    _,gnb=P.derive_selected_gnb(config)
    single=(ROOT/config['paths']['ue_source']).read_text()
    multi=ROOT/'OAI/openairinterface5g/targets/PROJECTS/GENERIC-NR-5GC/CONF/ue.multi2.conf'
    ue=multi.read_text()
    require(uicc_block(single,0).strip()==uicc_block(ue,0).strip(),'retained UE0 subscriber drift')
    for pattern,replacement in ((r'N_RB_DL\s*=\s*106;', 'N_RB_DL = 273;'),
                                (r'rf_freq\s*=\s*3619200000L;', 'rf_freq = 3649260000L;'),
                                (r'serveraddr\s*=\s*"127\.0\.0\.1";',
                                 'serveraddr = "127.0.0.1";\n    options = ("chanmod");')):
        ue,count=re.subn(pattern,replacement,ue)
        require(count==2,'two-UE template shape drift: '+pattern)
    channel=(ROOT/config['paths']['channel_source']).read_text()
    channel,count=re.subn(r'noise_power_dB\s*=\s*[-+0-9.eE]+;', 'noise_power_dB = -50;',channel)
    require(count==3 and 'noise_power_dBFS' not in channel,'channel source drift')
    # Same established explicit-per-UE RFsim objects as multiue_oai.runner;
    # the retained single-UE 273-PRB carrier only contains UL ue0. Duplicate
    # that AWGN object verbatim for ue1, with no calibration/parameter change.
    match=re.search(r'\{[^{}]*model_name\s*=\s*"rfsimu_channel_ue0"[^{}]*\}',channel)
    require(match is not None and channel.count('"rfsimu_channel_ue0"')==1
            and '"rfsimu_channel_ue1"' not in channel,'single-UE channel template drift')
    channel=channel[:match.end()]+',\n'+match.group(0).replace('rfsimu_channel_ue0','rfsimu_channel_ue1')+channel[match.end():]
    include='@include "channelmod_rfsimu_LEO_satellite.conf"'
    require(ue.count(include)==1,'UE include drift')
    ue=ue.replace(include,channel)
    validate_ue_pair(ue)
    outputs={}
    for name,text in (('gnb.conf',gnb+'\n'+channel),('ue.conf',ue)):
        path=Path(output)/name
        with path.open('x') as stream: stream.write(text)
        outputs[name]=sha(path)
    outputs['uicc1_source_sha256']=sha(multi)
    return outputs


def argv(preparation):
    ran=ROOT/'OAI/openairinterface5g/cmake_targets/ran_build/build'
    common=['--rfsim','--rfsimulator.[0].options','chanmod','--T_stdout','2','--T_nowait',
            '--loader.shlibpath',str(ran)]
    return dict(
        gnb=['sudo','-n','env','-u','LD_PRELOAD','-u','SCENESENSE_FORCE_UL_MCS','-u','SCENESENSE_HOLD_MCS_FEW_SAMPLES',
             '-u','SCENESENSE_AIMD_MAX_DROP','LD_LIBRARY_PATH='+str(ran),'SCENESENSE_MCS_POLICY=sinr',str(ran/'nr-softmodem'),
             '-O',str(Path(preparation)/'gnb.conf'),'--gNBs.[0].min_rxtxtime','6',*common,
             '--telnetsrv','--telnetsrv.listenaddr','127.0.0.1','--telnetsrv.listenport','9090','--T_port','2021'],
        ue=['sudo','-n','env','-u','LD_PRELOAD','LD_LIBRARY_PATH='+str(ran),str(ran/'nr-uesoftmodem'),'--num-ues','2',*common,
            '--rfsimulator.[0].serveraddr','127.0.0.1','-r','273','--numerology','1','--band','78',
            '-C','3649260000','--ssb','516','-O',str(Path(preparation)/'ue.conf'),'--T_port','2023'])


def prepare_workdirs(output):
    """Keep all OAI relative runtime writes outside the read-only build tree.

    Verify writes using the same sudo identity before starting either radio.
    Keep the tiny probe files as evidence; no historical stats are overwritten.
    Absolute binary/config paths and both loader paths preserve library binding.
    """
    base=Path(output)/'radio_work'
    base.mkdir()
    paths={role:base/role for role in ('gnb','ue')}
    for path in paths.values(): path.mkdir()
    probe=("import json,os,pathlib,sys; rows=[]; "
           "\nfor value in sys.argv[1:]:"
           "\n p=pathlib.Path(value); marker=p/'WRITE_PROBE';"
           "\n with marker.open('xb') as f: f.write(b'M4 radio runtime write verified\\n'); f.flush(); os.fsync(f.fileno())"
           "\n rows.append(dict(directory=str(p),euid=os.geteuid(),probe=str(marker),bytes=marker.stat().st_size))"
           "\nprint(json.dumps(rows))")
    result=subprocess.run(['sudo','-n','/usr/bin/python3','-B','-c',probe,
                           *map(str,paths.values())],capture_output=True,text=True,timeout=20)
    write_new(base/'WRITE_CHECK.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
    require(result.returncode==0,'radio runtime directories are not writable: '+result.stderr)
    rows=json.loads(result.stdout)
    require(len(rows)==2 and all(r['euid']==0 and r['bytes']>0 for r in rows),
            'radio runtime write identity/check differs')
    return paths


def wait_listener(port, children, timeout=60):
    """Wait for a local listener without consuming an OAI tracer connection."""
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        require(all(child.poll() is None for child in children),'radio exited during listener startup')
        for table in ('/proc/net/tcp','/proc/net/tcp6'):
            for row in Path(table).read_text().splitlines()[1:]:
                fields=row.split()
                if fields[3]=='0A' and int(fields[1].split(':')[1],16)==port: return
        time.sleep(.1)
    raise TimeoutError('radio listener not ready: '+str(port))


class AttachedRadio:
    def __init__(self, *, preparation, output, owner, config):
        self.preparation,self.output,self.owner,self.config=Path(preparation),Path(output),owner,config
        self.sources=None
        self.snr=None
        self.telemetry=None

    def start(self):
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import ue_telemetry_provider_v2 as T
        from rl_agent.splitfusion_m3_route_b_v20.radio_telemetry import RadioTelemetry
        commands=argv(self.preparation)
        workdirs=prepare_workdirs(self.output)
        gnb=self.owner.start('gnb',commands['gnb'],cwd=workdirs['gnb'],cpus=CPU['radio'])
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            require(gnb.poll() is None,'gNB exited before UE start'); time.sleep(.1)
        ue=self.owner.start('ue',commands['ue'],cwd=workdirs['ue'],cpus=CPU['radio'])
        tracer=ROOT/'OAI/openairinterface5g/common/utils/T/tracer'
        messages=tracer.parent/'T_messages.txt'
        wait_listener(2023,[gnb,ue])
        relay=self.owner.start('ue_relay',[str(tracer/'multi'),'-d',str(messages),'-ip','127.0.0.1',
                                    '-p','2023','-lp','2123'],cwd=tracer,cpus=CPU['radio'])
        wait_listener(2123,[gnb,ue,relay])
        discovery=self.owner.start('attach_rlc',T.csv_reader_argv(tracer,messages,2123,'NRUE_MAC_RLC_BUFFER_STATUS'),
                                   cwd=tracer,cpus=CPU['radio'])
        sessions={str(i):str(uuid.uuid4()) for i in (0,1)}
        deadline=time.monotonic()+120
        last=None
        while time.monotonic()<deadline:
            require(gnb.poll() is None and ue.poll() is None and discovery.poll() is None,'radio process exited')
            try:
                text=(self.output/'attach_rlc.log').read_text()
                header=','.join(T.EVENT_FIELDS['NRUE_MAC_RLC_BUFFER_STATUS'])
                require(header in text,'RLC attach header not ready')
                # A live CSV writer may be midway through the final line.
                complete=text[:text.rfind('\n')+1]
                rows=list(csv.DictReader(complete[complete.index(header):].splitlines()))
                self.sources=resolve(expected=list(EXPECTED),run_id=self.config['run_id'],cell_id=self.config['cell_id'],
                    attach_sessions=sessions,uicc_text=(self.preparation/'ue.conf').read_text(),
                    ue_log=(self.output/'ue.log').read_text(),interfaces=discover_interfaces(),rlc_rows=rows)
                break
            except (ValueError,subprocess.CalledProcessError,FileNotFoundError) as error:
                last=repr(error); time.sleep(.1)
        require(self.sources is not None,'two-source attach failed: '+str(last))
        write_new(self.output/'ATTACHED.json',dict(sources=[s.tag for s in self.sources.sources],
            discovery='actual UICC initialization order + tunnel leases + unambiguous UE/RNTI RLC pairs'))
        self.telemetry=RadioTelemetry(self.output/'pusch',CPU['radio'])
        self.telemetry.start()
        return self.sources

    def stop(self):
        try:
            if self.snr is not None: self.snr.close()
        finally:
            if self.telemetry is not None: self.telemetry.stop()


class GoodAll:
    """One 100-ms actuator owner; two separately ACKed native SNR publishers."""
    def __init__(self,sources,providers,output):
        from rl_agent import ue_target_snr_cell_runtime_v1 as B
        from rl_agent.splitfusion_run5_live_avo_v1 import snr_live as S
        self.B=B; self.oai=B.load_oai_module()
        self.session=self.oai.TelnetSession('127.0.0.1',9090,2.,1<<20)
        self.sources,self.providers=sources,providers
        self.publishers={s.stream_id:S.ControllerPublisher(providers[s.stream_id].path,
            providers[s.stream_id].controller_id,audit_path=Path(output)/(s.stream_id+'_ACKS.jsonl')) for s in sources.sources}
        self.indices={}; self.stop=threading.Event(); self.error=None
        self.log=(Path(output)/'COMMANDS.jsonl').open('x')
        try:
            *_,text=self.session.command('channelmod show current')
            models=self.oai.parse_channel_models(text)
            for s in sources.sources:
                row=models.get('rfsimu_channel_ue'+str(s.ue_id))
                require(row is not None and float(row['noise_power_db'])==-50.,'source channel missing/not clean')
                require(row.get('owner')=='rfsimulator' and row.get('model_type')=='AWGN'
                        and float(row['path_loss_db'])==0.,'RFsim ownership/type/path-loss drift')
                self.indices[s.stream_id]=int(row['model_index'])
            require(len(set(self.indices.values()))==2,'both UEs resolve to the same channel object')
        except BaseException:
            for publisher in self.publishers.values(): publisher.close()
            self.session.close(); self.log.close(); raise
        self.thread=threading.Thread(target=self.run,name='m4-good-all',daemon=True)
        self.thread.start()

    def run(self):
        try:
            tick=time.monotonic()
            while not self.stop.is_set():
                tick+=.1
                for source in self.sources.sources:
                    command=f'channelmod modify {self.indices[source.stream_id]} noise_power_dB -12.0'
                    sent_mono,sent_wall,ack_mono,ack_wall,response=self.session.command(command)
                    self.B.validate_modify(response,'-12.0',self.oai)
                    row=dict(source=source.tag,target_snr_db=23.5,mapped_rfsim_command_db=-12.,
                        command=command,command_ack_monotonic_ns=ack_mono,command_ack_wall_ns=ack_wall,
                        command_timing_status='ACK_ON_TIME' if time.monotonic()<tick else 'ACK_LATE')
                    self.log.write(json.dumps(row,sort_keys=True)+'\n'); self.log.flush()
                    self.publishers[source.stream_id].on_row(row)
                self.stop.wait(max(0.,tick-time.monotonic()))
                if time.monotonic()>tick+.1: tick=time.monotonic()  # no burst catch-up
        except BaseException as error:
            self.error=repr(error)
            self.stop.set()

    def close(self):
        self.stop.set(); self.thread.join(timeout=10)
        errors=[]
        try:
            for source in self.sources.sources:
                try:
                    *_,response=self.session.command(f'channelmod modify {self.indices[source.stream_id]} noise_power_dB -50.0')
                    self.B.validate_modify(response,'-50.0',self.oai)
                except BaseException as error: errors.append(repr(error))
                self.publishers[source.stream_id].close()
            *_,text=self.session.command('channelmod show current')
            models=self.oai.parse_channel_models(text)
            require(all(float(models['rfsimu_channel_ue'+str(s.ue_id)]['noise_power_db'])==-50. for s in self.sources.sources),
                    'two-source clean restoration failed')
            require(not errors,'radio restore errors: '+str(errors))
        finally:
            self.session.close(); self.log.close()
