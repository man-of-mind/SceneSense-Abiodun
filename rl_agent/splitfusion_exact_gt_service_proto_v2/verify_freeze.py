"""Step 2: binary source handoff equals the existing V4 staged codec on all 49 v11 frames.

Per frame: decode the existing staged packet (V4) and the new sealed binary freeze,
compare every decoded field exactly, then compute GT from the binary-decoded
source inside the mapping (which must release cleanly: no retained views) and
compare every output with the deployed runtime producer.

    cd abiodun && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 /usr/bin/python3 -m rl_agent.splitfusion_exact_gt_service_proto_v2.verify_freeze --name step2_freeze_01
"""
import argparse
import json
import os
import secrets

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_numeric_gt_runtime_v1 import staging as ST

from . import parity_common as P, freeze_codec as FZ, source_compare as SCMP
from .verify_deferred import comparable


def run(name):
    P.environment_ok()
    corpus = P.Corpus(49)
    output = P.create_output(name)
    session = secrets.token_hex(32)
    P.write_json(output / "PLAN.json", dict(schema="splitfusion.exact_gt_service_proto.step2_plan.v1",
        frames=len(corpus.cases), run_binding=corpus.binding, source_sha256=corpus.sources,
        reference_codec="runtime staging.encode_stage/decode_stage (unchanged V4 snapshot)",
        candidate_codec="freeze_codec: sealed memfd of aligned raw buffers + canonical descriptor; FreezeGate admission",
        downstream="DeferredProducerEngine on binary-decoded source inside the sealed mapping; compared with runtime producer",
        timing_recorded=False, live=False, gpu=False))
    rows = []
    try:
        ref = P.ReferenceDriver(corpus)
        ref_rows = {c.frame_id: ref.run(c) for c in corpus.cases}
        ref.close()
        cand = P.DeferredDriver(corpus)
        gate = FZ.FreezeGate(session)
        for sequence, case in enumerate(corpus.cases, start=1):
            packet, extras = corpus.staged_packet(case)
            v4 = ST.decode_stage(packet, extras)
            descriptor, chunks = FZ.freeze(corpus.sourcekwargs(case), case.kwargs["policy_timestamp_s"],
                                           session_id=session, sequence=sequence)
            fd, size = FZ.seal(chunks)
            try:
                with gate.admit(descriptor, fd) as admitted:
                    fields = SCMP.compare(v4, admitted.kwargs)
                    handle = cand.engine.prepare_unbound(admitted.kwargs,
                        population=case.evidence["dynamic_snapshot_evidence"], **corpus.offline_inputs(case))
                released_cleanly = True
            finally:
                os.close(fd)
            del v4
            prepared = cand.engine.bind(handle, corpus.plan(case), frame_context=corpus.context(case))
            committed = cand.engine.confirm_send(prepared.token, corpus.plan(case))
            ready = cand.engine.publish_ready(prepared.token)
            retired = cand.retire(case, prepared)
            row = dict(retired, history_before=ref_rows[case.frame_id]["history_before"],
                       identity_bytes=prepared.identity_bytes, semantic_sha256=prepared.semantic_sha256,
                       commit=committed, packet=ready, history_after=P.history(cand.engine))
            a, b = comparable(ref_rows[case.frame_id]), comparable(row)
            gt_exact = {k: a[k] == b[k] for k in a if k != "history_before"}
            ok = all(fields.values()) and all(gt_exact.values()) and released_cleanly
            rows.append(dict(frame_id=case.frame_id, sequence=sequence, decoded_fields_exact=fields,
                             gt_outputs_exact_vs_runtime=gt_exact, mapping_released_without_leaked_views=released_cleanly,
                             descriptor_bytes=len(descriptor), sealed_bytes=size, staged_v4_packet_bytes=len(packet),
                             chunks=len(chunks), ok=ok))
            if not ok:
                raise AssertionError("step 2 parity failure at frame %d" % case.frame_id)
        cand.close()
        result = dict(schema="splitfusion.exact_gt_service_proto.step2_result.v1",
            status="STEP2_BINARY_HANDOFF_EXACT", frames=len(rows), all_exact=all(r["ok"] for r in rows),
            decoded_fields=sorted(rows[0]["decoded_fields_exact"]),
            gt_fields=sorted(rows[0]["gt_outputs_exact_vs_runtime"]),
            gate_admitted=gate.admitted, gate_failure=gate.failure,
            sizes=dict(descriptor_bytes_max=max(r["descriptor_bytes"] for r in rows),
                       sealed_bytes_max=max(r["sealed_bytes"] for r in rows),
                       staged_v4_packet_bytes_max=max(r["staged_v4_packet_bytes"] for r in rows)),
            timing_recorded=False, performance_claim=False, live_qualified=False)
        P.write_json(output / "RESULT.json", result)
        print(json.dumps(result, indent=1))
    except BaseException as error:
        P.write_json(output / "FAILURE.json", dict(error=f"{type(error).__name__}: {error}", completed=len(rows),
                                                   last=rows[-1] if rows else None))
        raise
    finally:
        with (output / "FRAMES.jsonl").open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        P.seal_output(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    run(parser.parse_args().name)
