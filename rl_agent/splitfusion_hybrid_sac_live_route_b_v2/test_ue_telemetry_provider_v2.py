"""Phase-2B offline tests for the bounded live UE telemetry provider.

Synthetic ``csv -f`` lines and a controllable fake host clock only.  No OAI,
CARLA, CUDA, model or network process is started (one test spawns a trivial
Python child to prove EOF/death handling).
"""

from __future__ import annotations

import sys
import time
import unittest
import uuid

from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as contract
from rl_agent.ue_mcs_backlog_calibration_v1 import contract as V3C

from . import ue_telemetry_provider_v2 as T

UTC_OFFSET_S = -7 * 3600
REAL_TO_MONO = -1_789_000_000_000_000_000   # mono = real + this
MONO_TO_RAW = -12_345_678                   # raw = mono + this
DAY_NS = 86_400 * 1_000_000_000


class FakeHost:
    def __init__(self, real_ns: int) -> None:
        self.real_ns = real_ns
        self.real_to_mono = REAL_TO_MONO
        self.mono_to_raw = MONO_TO_RAW

    def pair(self) -> T.HostClockPairV1:
        mono = self.real_ns + self.real_to_mono
        return T.HostClockPairV1(real_ns=self.real_ns, mono_ns=mono,
                                 raw_ns=mono + self.mono_to_raw, spread_ns=100,
                                 utc_offset_s=UTC_OFFSET_S)

    def raw(self) -> int:
        return self.pair().raw_ns

    def advance(self, ns: int) -> None:
        self.real_ns += ns

    def tod(self, real_ns: int) -> str:
        us = (real_ns // 1000 + UTC_OFFSET_S * 1_000_000) % T.DAY_US
        seconds, micros = divmod(us, 1_000_000)
        hours, rest = divmod(seconds, 3600)
        minutes, secs = divmod(rest, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{micros:06d}"


RNTI, UE = 55228, 0


class Harness:
    def __init__(self, *, start_real_ns: int = 1_790_632_292_000_000_000,
                 warm: bool = True, bind: bool = True) -> None:
        self.host = FakeHost(start_real_ns)
        self.bridge = T.CausalClockBridgeV2(min_anchors=8, window=32)
        self.provider = T.UeTelemetryProviderV2(bridge=self.bridge)
        self.readers = {
            name: T.LiveEventReaderV2(name, handler, self.provider,
                                      clock=self.host.pair)
            for name, handler in (
                ("NRUE_MAC_DCI_GRANT", self.provider.on_dci),
                ("NRUE_MAC_RLC_BUFFER_STATUS", self.provider.on_rlc),
                ("NR_PDCP_TX_SDU", self.provider.on_pdcp),
            )
        }
        for name, reader in self.readers.items():
            reader.feed("connecting to 127.0.0.1:2123")
            reader.feed(f"turning ON {name}")
            reader.feed(",".join(T.EVENT_FIELDS[name]))
        if bind:
            self.provider.bind_ue(rnti=RNTI, oai_ue_id=UE)
        if warm:
            for _ in range(8):
                self.pdcp()

    # -- line builders; events are stamped `lag` ns before receipt --------------
    def pdcp(self, *, lag: int = 200_000) -> None:
        event_real = self.host.real_ns - lag
        mono = event_real + self.host.real_to_mono
        self.readers["NR_PDCP_TX_SDU"].feed(
            f"{self.host.tod(event_real)},{mono // 10**9},{mono % 10**9},0,1,1229")
        self.host.advance(1_000_000)

    def dci(self, *, mcs=20, table=0, rnd=0, ndi=1, direction=1, rnti=RNTI,
            lag=300_000, advance=500_000) -> None:
        event_real = self.host.real_ns - lag
        self.readers["NRUE_MAC_DCI_GRANT"].feed(
            f"{self.host.tod(event_real)},{direction},7,0,{rnti},499,0,499,6,"
            f"{mcs},{table},0,5,0,13,480,1,{ndi},0,{rnd},6,9480,1,2,2")
        self.host.advance(advance)

    def rlc(self, *, frame, slot, lcid, bytes_, rnti=RNTI, ue=UE, lag=300_000,
            advance=100_000) -> None:
        event_real = self.host.real_ns - lag
        self.readers["NRUE_MAC_RLC_BUFFER_STATUS"].feed(
            f"{self.host.tod(event_real)},{rnti},{ue},{frame},{slot},{lcid},0,"
            f"{bytes_},2147483647,-1,1")
        self.host.advance(advance)

    def identity(self, seq: int = 0, *, session=None, ue=None):
        return contract.DecisionIdentityV1(
            session_uuid=session or self.provider.session_uuid,
            ue_id=ue or self.provider.ue_label or "unbound", decision_seq=seq)

    def decide(self, *, seq=0, session=None, ue=None, enqueue_lead_ns=5_000_000):
        identity = self.identity(seq, session=session, ue=ue)
        snapshot, boundary, latency = T.open_decision(
            self.provider, identity, clock=self._ticking_clock())
        return snapshot, T.assemble_radio_evidence(
            snapshot, identity=identity, boundary=boundary,
            payload_enqueue_timestamp_ns=(
                boundary.action_open_timestamp_ns + enqueue_lead_ns))

    def _ticking_clock(self):
        def clock() -> int:
            self.host.advance(1_000)
            return self.host.raw()
        return clock


def _valid_tick(h: Harness, frame: int, slot: int, values=(100, 0, 250)) -> None:
    for lcid, value in zip((1, 2, 4), values):
        h.rlc(frame=frame, slot=slot, lcid=lcid, bytes_=value)


class SchemaAndClockTest(unittest.TestCase):
    def test_headers_match_registered_calibration_contract(self) -> None:
        self.assertEqual(T.DCI_FIELDS, V3C.DCI_GRANT_HEADER)
        self.assertEqual(T.RLC_FIELDS, V3C.RLC_BUFFER_HEADER)
        self.assertEqual(T.PDCP_FIELDS, V3C.PDCP_TX_SDU_HEADER)

    def test_training_freshness_is_exact(self) -> None:
        self.assertEqual(T.TRAINING_FRESHNESS.canonical_sha256(),
                         T.TRAINING_FRESHNESS_SHA256)
        self.assertEqual(T.TRAINING_FRESHNESS.prior_ul_mcs_max_age_ns, 100_000_000)

    def test_time_of_day_reconstruction(self) -> None:
        host = FakeHost(1_790_632_292_123_456_000)
        event = host.real_ns - 700_000_000
        self.assertEqual(
            T.reconstruct_event_realtime_ns(host.tod(event), host.pair()),
            event // 1000 * 1000)

    def test_midnight_crossing(self) -> None:
        midnight_utc = (1_790_632_292 // 86_400 + 1) * 86_400 - UTC_OFFSET_S
        host = FakeHost(midnight_utc * 10**9 + 150_000)  # 00:00:00.000150 local
        event = host.real_ns - 400_000                    # 23:59:59.999750 local
        self.assertTrue(host.tod(event).startswith("23:59:59"))
        self.assertTrue(host.tod(host.real_ns).startswith("00:00:00"))
        self.assertEqual(
            T.reconstruct_event_realtime_ns(host.tod(event), host.pair()),
            event // 1000 * 1000)

    def test_future_and_malformed_times_rejected(self) -> None:
        host = FakeHost(1_790_632_292_000_000_000)
        with self.assertRaises(ValueError):
            T.reconstruct_event_realtime_ns(host.tod(host.real_ns + 5_000), host.pair())
        with self.assertRaises(ValueError):
            T.reconstruct_event_realtime_ns(host.tod(host.real_ns - 3 * 10**9), host.pair())
        for bad in ("14:51:32.44748", "24:00:00.000000", "x", "14:51:32"):
            with self.assertRaises(ValueError):
                T.parse_time_of_day_us(bad)


class BridgeTest(unittest.TestCase):
    def test_warm_up_required(self) -> None:
        h = Harness(warm=False)
        for _ in range(7):
            h.pdcp()
        self.assertFalse(h.bridge.warm)
        h.dci()
        self.assertEqual(h.provider.counters["dci_rejected_bridge_not_warm"], 1)
        _, evidence = h.decide()
        self.assertIn("CLOCK_BRIDGE_INVALID_OR_WARMING", evidence.fallback_reasons)
        h.pdcp()
        self.assertTrue(h.bridge.warm)

    def test_conversion_is_exact_and_residual_small(self) -> None:
        h = Harness()
        for _ in range(20):
            h.pdcp()
        real = h.host.real_ns - 123_000
        expected = real // 1000 * 1000 + REAL_TO_MONO + MONO_TO_RAW
        self.assertEqual(h.bridge.to_raw(real // 1000 * 1000), expected)
        self.assertTrue(h.bridge.residuals_ns)
        self.assertLessEqual(max(abs(r) for r in h.bridge.residuals_ns), 1_000)

    def test_realtime_step_resets_and_clears_caches(self) -> None:
        h = Harness()
        h.dci()
        self.assertEqual(len(h.provider.snapshot().dci), 1)
        h.host.real_to_mono += 5_000_000            # REALTIME stepped by 5 ms
        h.pdcp()
        self.assertGreater(h.bridge.counters["reset_realtime_step"], 0)
        self.assertFalse(h.bridge.warm)
        self.assertEqual(h.provider.snapshot().dci, ())

    def test_monotonic_raw_jump_resets(self) -> None:
        h = Harness()
        h.host.mono_to_raw += 2_000_000
        h.pdcp()
        self.assertGreater(h.bridge.counters["reset_monotonic_raw_jump"], 0)
        self.assertFalse(h.bridge.warm)

    def test_anchor_discontinuity_resets(self) -> None:
        h = Harness()
        # An anchor whose own offset jumps (UE-side clock anomaly) while the
        # host pairs stay continuous.
        event_real = h.host.real_ns - 200_000
        mono = event_real + REAL_TO_MONO - 3_000_000
        h.provider.on_pdcp({"time": h.host.tod(event_real),
                            "mono_sec": mono // 10**9, "mono_nsec": mono % 10**9},
                           h.host.pair())
        self.assertEqual(h.bridge.counters["reset_anchor_discontinuity"], 1)
        self.assertFalse(h.bridge.warm)

    def test_future_monotonic_anchor_rejected(self) -> None:
        h = Harness()
        event_real = h.host.real_ns - 200_000
        mono = h.host.pair().mono_ns + 50_000
        h.provider.on_pdcp({"time": h.host.tod(event_real),
                            "mono_sec": mono // 10**9, "mono_nsec": mono % 10**9},
                           h.host.pair())
        self.assertEqual(h.provider.counters["pdcp_rejected_future_mono"], 1)


class DciTest(unittest.TestCase):
    def test_ndi_zero_and_one_both_accepted(self) -> None:
        h = Harness()
        _valid_tick(h, 10, 1)
        _valid_tick(h, 10, 2)
        h.dci(mcs=17, ndi=0)
        _, evidence = h.decide()
        self.assertTrue(evidence.admitted, evidence.fallback_reasons)
        self.assertEqual(evidence.prior_ul_mcs.observation.value, 17)
        self.assertEqual(evidence.prior_ul_mcs.new_data_indicator, 0)
        h.dci(mcs=19, ndi=1)
        _valid_tick(h, 10, 3)
        _, evidence = h.decide(seq=1)
        self.assertEqual(evidence.prior_ul_mcs.observation.value, 19)
        self.assertEqual(evidence.prior_ul_mcs.new_data_indicator, 1)

    def test_rejected_classes_never_enter_cache(self) -> None:
        h = Harness()
        h.dci(mcs=11)
        for kwargs, counter in (
            ({"rnd": 1}, "dci_rejected_retransmission_round"),
            ({"direction": 0}, "dci_rejected_direction"),
            ({"table": 1}, "dci_rejected_table"),
            ({"rnti": RNTI + 1}, "dci_rejected_cross_ue"),
            ({"ndi": 2}, "dci_malformed_ndi"),
        ):
            h.dci(mcs=3, **kwargs)
            self.assertEqual(h.provider.counters[counter], 1, counter)
        self.assertEqual([c.mcs_index for c in h.provider.snapshot().dci], [11])

    def test_out_of_range_mcs_fails_closed(self) -> None:
        h = Harness()
        _valid_tick(h, 1, 1)
        _valid_tick(h, 1, 2)
        h.dci(mcs=29)
        _, evidence = h.decide()
        self.assertFalse(evidence.admitted)
        self.assertTrue(any(r.startswith("UL_MCS_SELECTOR_REFUSED")
                            for r in evidence.fallback_reasons))
        self.assertFalse(evidence.prior_ul_mcs.observation.metadata.valid)

    def test_mcs_zero_is_valid(self) -> None:
        h = Harness()
        _valid_tick(h, 1, 1)
        _valid_tick(h, 1, 2)
        h.dci(mcs=0)
        _, evidence = h.decide()
        self.assertTrue(evidence.admitted, evidence.fallback_reasons)
        self.assertEqual(evidence.prior_ul_mcs.observation.value, 0)

    def test_unbound_provider_rejects(self) -> None:
        h = Harness(bind=False)
        h.dci()
        self.assertEqual(h.provider.counters["dci_unbound"], 1)
        _, evidence = h.decide(ue="oai-ue0-rnti55228")
        self.assertIn("CROSS_UE_OR_UNBOUND_TELEMETRY", evidence.fallback_reasons)


class RlcTest(unittest.TestCase):
    def test_multi_lcid_aggregation_publishes_only_complete_ticks(self) -> None:
        h = Harness()
        _valid_tick(h, 5, 1, values=(100, 20, 3))
        self.assertEqual(h.provider.snapshot().rlc, ())   # open tick hidden
        h.rlc(frame=5, slot=2, lcid=1, bytes_=7)
        rlc = h.provider.snapshot().rlc
        self.assertEqual([s.backlog_bytes for s in rlc], [123])
        self.assertIn("lcids=3", rlc[0].source)

    def test_duplicate_and_contradictory_rows(self) -> None:
        h = Harness()
        h.rlc(frame=5, slot=1, lcid=4, bytes_=10)
        h.rlc(frame=5, slot=1, lcid=4, bytes_=10)
        h.rlc(frame=5, slot=1, lcid=5, bytes_=1)
        h.rlc(frame=5, slot=1, lcid=5, bytes_=2)
        h.rlc(frame=5, slot=2, lcid=4, bytes_=0)
        self.assertEqual(h.provider.counters["rlc_duplicate_lcid_row"], 1)
        self.assertEqual(
            h.provider.counters["rlc_contradictory_lcid_row_latest_wins"], 1)
        self.assertEqual(h.provider.snapshot().rlc[-1].backlog_bytes, 12)

    def test_availability_is_the_completion_proof_instant(self) -> None:
        h = Harness()
        _valid_tick(h, 5, 1)
        proof_raw = h.host.raw()
        h.rlc(frame=5, slot=2, lcid=1, bytes_=0)
        sample = h.provider.snapshot().rlc[-1]
        self.assertEqual(sample.available_timestamp_ns, proof_raw)
        self.assertLess(sample.source_timestamp_ns, sample.available_timestamp_ns)

    def test_measured_zero_is_valid_and_distinct_from_missing(self) -> None:
        h = Harness()
        h.dci()
        _valid_tick(h, 5, 1, values=(0, 0, 0))
        _valid_tick(h, 5, 2, values=(0, 0, 0))
        _, evidence = h.decide()
        self.assertTrue(evidence.admitted, evidence.fallback_reasons)
        self.assertEqual(evidence.pre_action_rlc_backlog.value, 0)
        h2 = Harness()
        h2.dci()
        _, missing = h2.decide()
        self.assertFalse(missing.admitted)
        self.assertIsNone(missing.pre_action_rlc_backlog.value)
        self.assertTrue(any("MISSING" in r for r in missing.fallback_reasons))

    def test_cross_ue_rlc_rows_ignored_and_do_not_close_ticks(self) -> None:
        h = Harness()
        _valid_tick(h, 5, 1)
        h.rlc(frame=5, slot=2, lcid=1, bytes_=9, ue=3)
        self.assertEqual(h.provider.snapshot().rlc, ())
        self.assertEqual(h.provider.counters["rlc_rejected_cross_ue"], 1)

    def test_negative_bytes_malformed(self) -> None:
        h = Harness()
        h.rlc(frame=5, slot=1, lcid=1, bytes_=-1)
        self.assertEqual(h.provider.counters["rlc_malformed_negative_bytes"], 1)


class DecisionTest(unittest.TestCase):
    def _ready(self) -> Harness:
        h = Harness()
        h.dci(mcs=21)
        _valid_tick(h, 7, 1)
        _valid_tick(h, 7, 2)
        return h

    def test_admitted_ordering_invariants(self) -> None:
        h = self._ready()
        _, evidence = h.decide()
        self.assertTrue(evidence.admitted, evidence.fallback_reasons)
        b = evidence.boundary
        for obs in (evidence.prior_ul_mcs.observation, evidence.pre_action_rlc_backlog):
            m = obs.metadata
            self.assertLessEqual(m.source_timestamp_ns, m.available_timestamp_ns)
            self.assertLess(m.available_timestamp_ns, b.state_commit_timestamp_ns)
            self.assertEqual(m.clock_domain, "CLOCK_MONOTONIC_RAW")
        self.assertLess(b.state_commit_timestamp_ns, b.action_open_timestamp_ns)
        self.assertLess(b.action_open_timestamp_ns,
                        evidence.payload_enqueue_timestamp_ns)

    def test_stale_samples_fall_back(self) -> None:
        h = self._ready()
        h.host.advance(101_000_000)
        _, evidence = h.decide()
        self.assertFalse(evidence.admitted)
        self.assertTrue(any("STALE" in r for r in evidence.fallback_reasons))

    def test_cross_session_and_cross_ue(self) -> None:
        h = self._ready()
        _, evidence = h.decide(session=str(uuid.uuid4()))
        self.assertIn("CROSS_SESSION_TELEMETRY", evidence.fallback_reasons)
        _, evidence = h.decide(ue="oai-ue1-rnti1")
        self.assertIn("CROSS_UE_OR_UNBOUND_TELEMETRY", evidence.fallback_reasons)

    def test_cross_epoch_samples_are_ignored_by_selectors(self) -> None:
        old = self._ready()
        new = self._ready()
        new.host = old.host
        foreign = old.provider.snapshot()
        identity = new.identity()
        _, boundary, _ = T.open_decision(new.provider, identity,
                                         clock=new._ticking_clock())
        mixed = T.TelemetrySnapshotV2(
            seq=1, session_uuid=new.provider.session_uuid,
            ue_label=new.provider.ue_label, bridge_warm=True, bridge_generation=0,
            readers_alive=foreign.readers_alive, dci=foreign.dci, rlc=foreign.rlc)
        evidence = T.assemble_radio_evidence(
            mixed, identity=identity, boundary=boundary,
            payload_enqueue_timestamp_ns=boundary.action_open_timestamp_ns + 1)
        self.assertFalse(evidence.admitted)
        self.assertFalse(evidence.prior_ul_mcs.observation.metadata.valid)

    def test_no_future_sample_enters_a_decision(self) -> None:
        h = self._ready()
        identity = h.identity()
        snapshot, boundary, _ = T.open_decision(h.provider, identity,
                                                clock=h._ticking_clock())
        h.dci(mcs=5)                                   # arrives after commit
        _valid_tick(h, 7, 3)
        self.assertNotEqual(h.provider.snapshot().seq, snapshot.seq)
        evidence = T.assemble_radio_evidence(
            h.provider.snapshot(), identity=identity, boundary=boundary,
            payload_enqueue_timestamp_ns=boundary.action_open_timestamp_ns + 1)
        self.assertEqual(evidence.prior_ul_mcs.observation.value, 21)
        for obs in (evidence.prior_ul_mcs.observation, evidence.pre_action_rlc_backlog):
            self.assertLess(obs.metadata.available_timestamp_ns,
                            boundary.state_commit_timestamp_ns)

    def test_enqueue_must_follow_action_open(self) -> None:
        h = self._ready()
        identity = h.identity()
        snapshot, boundary, _ = T.open_decision(h.provider, identity,
                                                clock=h._ticking_clock())
        with self.assertRaises(contract.MetadataError):
            T.assemble_radio_evidence(
                snapshot, identity=identity, boundary=boundary,
                payload_enqueue_timestamp_ns=boundary.action_open_timestamp_ns)


class ReaderLifecycleTest(unittest.TestCase):
    def test_malformed_header_kills_reader(self) -> None:
        h = Harness(warm=False)
        bridge = T.CausalClockBridgeV2()
        provider = T.UeTelemetryProviderV2(bridge=bridge)
        reader = T.LiveEventReaderV2("NRUE_MAC_DCI_GRANT", provider.on_dci,
                                     provider, clock=h.host.pair)
        reader.feed("time,direction,wrong")
        self.assertEqual(reader.counters["malformed_header"], 1)
        self.assertFalse(dict(provider.snapshot().readers_alive)["NRUE_MAC_DCI_GRANT"])

    def test_malformed_rows_counted_not_fatal(self) -> None:
        h = Harness()
        reader = h.readers["NRUE_MAC_DCI_GRANT"]
        reader.feed("14:00:00.000000,1,2")
        reader.feed("bad-time," + ",".join(["1"] * 24))
        reader.feed("14:00:00.000000," + ",".join(["x"] * 24))
        self.assertEqual(reader.counters["malformed_row"], 3)
        self.assertTrue(h.provider.snapshot().all_readers_alive)

    def test_eof_marks_dead_and_forces_fallback(self) -> None:
        h = Harness()
        h.dci()
        _valid_tick(h, 1, 1)
        _valid_tick(h, 1, 2)
        h.readers["NRUE_MAC_RLC_BUFFER_STATUS"].eof()
        self.assertEqual(h.readers["NRUE_MAC_RLC_BUFFER_STATUS"]
                         .counters["unexpected_eof"], 1)
        _, evidence = h.decide()
        self.assertIn("TELEMETRY_READER_DEAD", evidence.fallback_reasons)

    def test_real_child_process_death(self) -> None:
        bridge = T.CausalClockBridgeV2()
        provider = T.UeTelemetryProviderV2(bridge=bridge)
        reader = T.LiveEventReaderV2("NR_PDCP_TX_SDU", provider.on_pdcp, provider)
        header = ",".join(T.PDCP_FIELDS)
        reader.start([sys.executable, "-c",
                      f"print('turning ON NR_PDCP_TX_SDU'); print({header!r})"],
                     cwd=T.Path.cwd())
        reader._thread.join(timeout=10)
        self.assertEqual(reader.counters["unexpected_eof"], 1)
        self.assertFalse(dict(provider.snapshot().readers_alive)["NR_PDCP_TX_SDU"])
        reader.stop()


class BoundednessAndFaultTest(unittest.TestCase):
    def test_cache_size_is_bounded(self) -> None:
        h = Harness()
        for index in range(2_000):
            h.dci(mcs=index % 29, advance=1_000)
            h.rlc(frame=index % 1024, slot=index % 20, lcid=1, bytes_=index,
                  advance=1_000)
        sizes = h.provider.cache_sizes()
        self.assertEqual(sizes["dci"], sizes["dci_capacity"])
        self.assertEqual(sizes["rlc"], sizes["rlc_capacity"])
        self.assertLessEqual(len(h.provider.snapshot().dci), 8)

    def test_snapshot_is_constant_time_reference_read(self) -> None:
        h = Harness()
        first = h.provider.snapshot()
        self.assertIs(first, h.provider.snapshot())
        timings = []
        for fill in (10, 5_000):
            for index in range(fill):
                h.dci(advance=1_000)
            start = time.perf_counter_ns()
            for _ in range(100_000):
                h.provider.snapshot()
            timings.append(time.perf_counter_ns() - start)
        self.assertLess(timings[1], timings[0] * 3 + 5_000_000)
        self.assertLess(timings[1] / 100_000, 1_000)        # << 1 us per call

    def test_actor_never_called_on_any_fault(self) -> None:
        calls = []

        def actor(_evidence):
            calls.append(1)
            return "ACTED"

        faults = []
        h = Harness(warm=False)
        faults.append(h.decide()[1])                         # bridge warming
        h = self_ready = DecisionTest._ready(DecisionTest())
        h.host.advance(150_000_000)
        faults.append(h.decide()[1])                         # stale
        h = DecisionTest._ready(DecisionTest())
        faults.append(h.decide(session=str(uuid.uuid4()))[1])  # cross-session
        faults.append(h.decide(ue="oai-ue9-rnti9")[1])       # cross-UE
        h.readers["NR_PDCP_TX_SDU"].eof()
        faults.append(h.decide()[1])                         # dead reader
        h = Harness()
        faults.append(h.decide()[1])                         # missing both
        h = Harness()
        _valid_tick(h, 1, 1)
        _valid_tick(h, 1, 2)
        h.dci(mcs=40)
        faults.append(h.decide()[1])                         # malformed MCS
        for evidence in faults:
            self.assertFalse(evidence.admitted)
            with self.assertRaises(contract.ExternalFallbackRequired):
                T.act_or_fallback(evidence, actor)
        self.assertEqual(calls, [])
        del self_ready
        ok = DecisionTest._ready(DecisionTest()).decide()[1]
        self.assertEqual(T.act_or_fallback(ok, actor), "ACTED")
        self.assertEqual(calls, [1])


if __name__ == "__main__":
    unittest.main()
