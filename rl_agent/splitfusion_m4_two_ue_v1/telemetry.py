"""Single established UE tracer relay, three real readers, strict per-UE fanout."""
from pathlib import Path
from .policy import TelemetryFanout
from .configuration import CPU, ROOT, write_new


class NativeTelemetry:
    def __init__(self, contexts, output):
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import ue_telemetry_provider_v2 as T
        self.fanout = TelemetryFanout(contexts)
        self.readers = []
        self.output = Path(output)
        self.audit = None
        class Health:
            def reader_alive(self, event, alive):
                for context in contexts.values(): context['telemetry'].reader_alive(event,alive)
        tracer = ROOT/'OAI/openairinterface5g/common/utils/T/tracer'
        try:
            for event in ('NRUE_MAC_DCI_GRANT','NRUE_MAC_RLC_BUFFER_STATUS','NR_PDCP_TX_SDU'):
                reader=T.LiveEventReaderV2(event,lambda row,pair,e=event:self.fanout.dispatch(e,row,pair),Health())
                self.readers.append(reader)
                reader.start(['taskset','-c',','.join(map(str,CPU['radio'])),
                    *T.csv_reader_argv(tracer,tracer.parent/'T_messages.txt',2123,event)],cwd=tracer)
            self.audit=T.AuditWriterV2(self.readers,self.output/'native_radio')
            self.audit.start()
        except BaseException:
            self.close(); raise
    def close(self):
        for reader in self.readers: reader.stop()
        if self.audit is not None: self.audit.stop()
        write_new(self.output/'TELEMETRY_CLOSED.json',dict(
            counters={r.event:dict(r.counters) for r in self.readers},rejected=self.fanout.rejected))
