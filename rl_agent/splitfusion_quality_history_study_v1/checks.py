"""Focused no-gradient checks, not another training run or broad qualification suite."""
import dataclasses as D
import json
from collections import Counter

import torch

from . import adapter as A
from .study import make_orchestrator, REFERENCE, read_json


def run():
    torch.set_num_threads(4)
    control = make_orchestrator('CONTROL')
    candidate = make_orchestrator('CANDIDATE')
    initial_diffs = {}
    for old, new, keys in ((control.actor, candidate.actor, (A.RM.ACTOR_INPUT_KEY,)),
                          (control.critics, candidate.critics, A.RM.CRITIC_INPUT_KEYS)):
        for k, v in old.state_dict().items():
            w = new.state_dict()[k]
            if k in keys:
                assert torch.equal(v, torch.cat((w[:, :22], w[:, 24:]), dim=1))
                assert not bool(w[:, 22:24].abs().sum())
            else:
                assert torch.equal(v, w)
    # Same original function even when new features are nonzero. Float32 widened
    # GEMM may differ by rounding; execution must remain exact in these states.
    g = torch.Generator().manual_seed(1717)
    x = torch.rand((64, 22), generator=g)
    y = torch.cat((x, torch.rand((64, 2), generator=g)), dim=1)
    with torch.no_grad():
        h, j = control.actor(x), candidate.actor(y)
        for name in ('logits', 'mean', 'log_std'):
            # ActorHeads names are asserted, not inferred from sampled actions.
            initial_diffs[name] = float((getattr(h, name)-getattr(j, name)).abs().max())
            assert torch.allclose(getattr(h, name), getattr(j, name), atol=2e-6, rtol=2e-6)
        a, b = control.actor.deterministic_execution(x), candidate.actor.deterministic_execution(y)
        assert torch.equal(a.mode_index, b.mode_index) and torch.equal(a.q_e4, b.q_e4)
        modes = torch.nn.functional.one_hot(torch.arange(64) % 12, num_classes=12).to(torch.float32)
        q = torch.rand((64,), generator=g)
        for name in ('critic_1', 'critic_2', 'target_1', 'target_2'):
            a, b = getattr(control.critics, name)(x, modes, q), getattr(candidate.critics, name)(y, modes, q)
            initial_diffs[name] = float((a-b).abs().max())
            assert torch.allclose(a, b, atol=2e-6, rtol=2e-6)
    z = (0.0,) * 22
    assert A.encode(z) == (0.0,) * 24
    token = ('environment', 'session', 0)
    h = A.UnscoredHistory(11, 7000, 100.0, 50, token)
    u = A.encode(z, unscored=h, token=token, commit_ns=50)
    assert u[15] == u[19] == 1 and u[16] == 7000/9800 and u[23] == 100/170
    assert u[17] == u[18] == u[20] == u[22] == 0
    # Repeated unavailable quality uses last action/receipt, never last success.
    h2 = A.UnscoredHistory(10, 4000, 80.0, 70, ('environment', 'new_session', 0))
    u2 = A.encode(z, unscored=h2, token=h2.successor_token, commit_ns=70)
    assert u2[14] == 1 and u2[15] == 0 and u2[16] == 4000/9800 and u2[23] == 80/170
    for changed in (('other_environment', 'session', 0), ('environment', 'other_session', 0),
                    ('environment', 'session', 1)):
        assert A.encode(z, unscored=h, token=changed, commit_ns=100) == (0.0,) * 24
    try:
        A.encode(z, unscored=h, token=token, commit_ns=49)
        raise AssertionError('future data accepted')
    except ValueError:
        pass
    success = list(z)
    success[4] = success[19] = success[20] = 1
    success[17], success[18] = .8, .6
    assert A.encode(success) == tuple(success) + (1., 0.)
    timeout = list(success)
    timeout[17] = timeout[18] = timeout[20] = 0
    assert A.encode(timeout) == tuple(timeout) + (0., 0.)
    # Real paired warm-up, no optimizer steps: control digest must match sealed u0.
    for o in (control, candidate):
        o.warmup_phase()
    sealed = read_json(REFERENCE / 'checkpoints/u00000.json')
    for k, v in control.state_digest().items():
        assert sealed[k] == v, ('control mismatch', k)
    types = Counter()
    replayed = 0
    for d, e in zip(control.collector.diagnostics, candidate.collector.diagnostics):
        for k in ('mode_id', 'q_e4', 'receipt_ms', 'reward', 'terminal', 'reward_scene_key',
                  'prior_ul_mcs', 'snr_db', 'wire_bytes', 'base_chain_id'):
            assert d[k] == e[k], ('warm-up environment drift', k)
        types[e['terminal']] += 1
        if e['legitimate_undefined']:
            assert e['excluded'] and e['reward'] is None and e['history_retained']
            bad = dict(e, terminal='INFRASTRUCTURE_FAULT')
            assert not A.legitimate_undefined(bad, candidate.collector.catalog)
            for altered in (dict(receipt_ms=171), dict(receipt_ms=float('nan')),
                            dict(reward=0.), dict(q_perc=0.), dict(excluded=False)):
                assert not A.legitimate_undefined(dict(e, **altered), candidate.collector.catalog)
            cat = candidate.collector.catalog
            defined = next(k for k in cat.keys if k in cat.defined_keys)
            assert not A.legitimate_undefined(dict(e, reward_scene_key=defined), cat)
            # Serialized evidence replay uses the same encoder, not state swaps.
            recorded = json.loads(json.dumps(dict(mode=e['mode_id'], q_e4=e['q_e4'],
                receipt_ms=e['receipt_ms'], available_ns=50, successor_token=token)))
            recorded['successor_token'] = tuple(recorded['successor_token'])
            replay = A.encode(z, unscored=A.UnscoredHistory(**recorded), token=token, commit_ns=50)
            assert replay[4:21] == tuple(e['candidate_next_state'][4:21])
            assert replay[22:] == tuple(e['candidate_next_state'][22:])
            replayed += 1
    assert replayed > 0 and types['SUCCESS'] and types['TIMEOUT']
    assert len(control.replay) == len(candidate.replay)
    assert all(r.terminal != 'EVALUATOR_FAULT' and r.reward is not None for r in candidate.replay._rows)
    candidate.trainer._preflight(candidate.replay.sample(32, torch.Generator().manual_seed(1)))
    # Genuine external session reset invalidates any pending history.
    candidate.collector.unscored = A.UnscoredHistory(11, 7000, 100, 0, candidate.collector._boundary()[0])
    candidate.collector.base._env = candidate.collector.base._build_env()
    candidate.collector.base._session_index += 1
    candidate.collector.base._open_session()
    assert not any(candidate.collector.current_state_features()[4:21])
    assert candidate.collector.unscored is None
    assert control.update_count == candidate.update_count == 0
    assert not torch.cuda.is_initialized()
    return dict(passed=True, no_gradient_steps=True, control_u0_digest_exact=True,
        fair_initialization_max_abs=initial_diffs, initial_action_parity=64,
        cases=['genesis','defined timely','legitimate undefined','timeout','genuine fault',
               'repeated undefined','future refusal','external boundary reset'],
        real_warmup_cases=dict(types), recorded_event_replays=replayed,
        excluded_missing_quality_rows=288-len(candidate.replay), cuda_initialized=False)
