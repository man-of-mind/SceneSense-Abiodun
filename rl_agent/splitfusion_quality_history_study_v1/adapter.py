"""Offline-only observation adapter. Original transport/reward/transition logic is untouched."""
from __future__ import annotations

import dataclasses as D
import hashlib
import json
import math
from itertools import chain

import torch

from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_models as RM, run5_training as RT
from rl_agent.splitfusion_hybrid_sac_run5_v1.run5_collector import Run5CollectedTransitionV1
from rl_agent.splitfusion_hybrid_sac_v1.hybrid_sac_models import (
    ConditionalHybridActor, TwinHybridCritics, build_actor, build_twin_critics)
from rl_agent.splitfusion_run5_reward_trials_v1.trainer import TrialOrchestratorV1

WIDTH = 24
SCHEMA = 'quality_history_offline_v1_24d'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def legitimate_undefined(d, catalog):
    """Not every EVALUATOR_FAULT is benign: require actual pinned GT evidence."""
    if not (d['terminal'] == 'EVALUATOR_FAULT' and d['excluded'] and
            d['q_perc'] is None and d['reward'] is None and d['reward_value'] is None):
        return False
    ms = d['receipt_ms']
    if ms is None or not math.isfinite(ms) or not 0 <= ms <= 170:
        return False
    entry = catalog._scenes[d['reward_scene_key']]
    for q, w in catalog._bracket(d['q_e4']):
        if w <= 0:
            continue
        cell = (d['mode_id'], q)
        if entry['grid'][cell][0] is not None:
            return False
        if any(entry['components'][cell][c]['eligible'] != 0 for c in ('vehicle', 'person')):
            return False
    return True


@D.dataclass(frozen=True)
class UnscoredHistory:
    mode: int
    q_e4: int
    receipt_ms: float
    available_ns: int
    successor_token: tuple


def encode(base, *, unscored=None, token=None, commit_ns=None):
    """Shared training/recorded-event encoder, never reads a current action/outcome.

    An identified bookkeeping reset is bridged ONLY at its exact successor.
    A different session/decision/environment token invalidates the overlay.
    """
    require(len(base) == 22, 'expected original 22 features')
    x = list(map(float, base)) + [float(base[20]), 0.0]
    if unscored is not None and token == unscored.successor_token:
        require(commit_ns is not None and unscored.available_ns <= commit_ns,
                'future unscored history refused')
        require(0 <= unscored.mode < 12 and 0 <= unscored.q_e4 <= 9800, 'invalid action')
        require(math.isfinite(unscored.receipt_ms) and 0 <= unscored.receipt_ms <= 170,
                'invalid timely receipt')
        require(not any(x[4:21]), 'unscored overlay requires the identified legacy reset')
        x[4 + unscored.mode] = 1.0
        x[16] = unscored.q_e4 / 9800.0
        x[19] = 1.0
        x[23] = unscored.receipt_ms / 170.0
    validate_features(x)
    return tuple(x)


def validate_features(x):
    require(len(x) == WIDTH and all(math.isfinite(v) for v in x), 'invalid candidate state')
    require(x[19] in (0, 1) and x[20] in (0, 1) and x[22] in (0, 1), 'invalid mask')
    require(x[22] == x[20] and x[20] <= x[19], 'quality mask/legacy success disagree')
    require(all(v in (0, 1) for v in x[4:16]) and sum(x[4:16]) == x[19], 'invalid one-hot')
    require(0 <= x[16] <= 1 and 0 <= x[23] <= 1, 'invalid normalized action/receipt')
    if not x[19]:
        require(not any(x[4:21]) and x[23] == 0, 'genesis contains stale history')
    if not x[20]:
        require(x[17] == x[18] == 0, 'unavailable quality/legacy latency must be masked')
    else:
        require(x[23] == 0, 'defined outcome must not populate unscored latency')
    require(x[23] == 0 or (x[19] == 1 and x[22] == 0), 'unscored receipt mask invalid')


@D.dataclass(frozen=True, slots=True)
class CandidateTransition(Run5CollectedTransitionV1):
    def __post_init__(self):
        validate_features(self.state)
        if self.next_state is not None:
            validate_features(self.next_state)
        # Reuse ALL inherited non-width record checks on the exact 22-D prefix.
        fields = {f.name: getattr(self, f.name) for f in D.fields(self)}
        fields.update(state=self.state[:22], next_state=None if self.next_state is None else self.next_state[:22])
        Run5CollectedTransitionV1(**fields)


class HistoryCollector:
    def __init__(self, base):
        self.base = base
        self.unscored = None
        self.collector_binding_sha256 = hashlib.sha256(json.dumps(
            dict(schema=SCHEMA, base=base.collector_binding_sha256), sort_keys=True).encode()).hexdigest()

    def __getattr__(self, name):
        return getattr(self.base, name)

    def _boundary(self):
        bundle = self.base._env.current_state
        identity = bundle.state.state.identity
        return ((id(self.base._env), identity.session_uuid, identity.decision_seq),
                bundle.state.boundary.state_commit_timestamp_ns)

    def current_state_features(self):
        token, commit = self._boundary()
        if self.unscored is not None and self.unscored.successor_token != token:
            self.unscored = None  # genuine session/environment boundary, no bridge
        return encode(self.base.current_state_features(), unscored=self.unscored,
                      token=token, commit_ns=commit)

    def collect_transition(self, request):
        state = self.current_state_features()
        context = self.base.context
        record = self.base.collect_transition(request)
        d = self.base.diagnostics[-1]
        benign = legitimate_undefined(d, self.base.catalog)
        self.unscored = None
        if benign:
            require(record is None, 'undefined reward must remain excluded')
            token, commit = self._boundary()
            require(context.resolution_ns <= commit, 'outcome unavailable at successor commit')
            self.unscored = UnscoredHistory(request.mode_id, request.q_e4, d['receipt_ms'],
                                           context.resolution_ns, token)
        successor = self.current_state_features()
        d.update(candidate_state=state, candidate_next_state=successor,
                 legitimate_undefined=benign, history_retained=benign and successor[19] == 1)
        if record is None:
            return None
        values = {f.name: getattr(record, f.name) for f in D.fields(record)}
        values.update(state=state, next_state=successor if record.has_next_state else None)
        return CandidateTransition(**values)


def model_config():
    return D.replace(RM.run5_model_config(), state_dim=WIDTH)


def extend_models(actor, critics):
    """Copy initial function; no random columns or shifted critic actions."""
    out = (build_actor(model_config(), seed=0), build_twin_critics(model_config(), seed=0))
    for old, new, keys in ((actor, out[0], (RM.ACTOR_INPUT_KEY,)),
                           (critics, out[1], RM.CRITIC_INPUT_KEYS)):
        state = {k: v.clone() for k, v in old.state_dict().items()}
        for k in keys:
            v = state[k]
            widened = torch.zeros((v.shape[0], v.shape[1] + 2), dtype=v.dtype)
            widened[:, :22] = v[:, :22]
            widened[:, 24:] = v[:, 22:]
            state[k] = widened
        new.load_state_dict(state, strict=True)
    validate_models(*out)
    return out


def validate_models(actor, critics):
    require(type(actor) is ConditionalHybridActor and type(critics) is TwinHybridCritics, 'foreign models')
    for m in (actor, critics.critic_1, critics.critic_2, critics.target_1, critics.target_2):
        require(m.config == model_config(), 'candidate configuration drift')
        require(all(t.device.type == 'cpu' for t in chain(m.parameters(), m.buffers())), 'CPU only')
    require(actor.encoder[0].in_features == WIDTH, 'actor input width')
    require(all(c.trunk[0].in_features == WIDTH + 13 for c in
                (critics.critic_1, critics.critic_2, critics.target_1, critics.target_2)), 'critic input width')


class Replay(RT.Run5ReplayBufferV1):
    def insert(self, record):
        require(type(record) is CandidateTransition, 'candidate replay accepts exact candidate records only')
        digest, key = record.digest, (record.session_uuid, record.decision_seq)
        require(digest not in self._seen_digests and key not in self._seen_identities, 'duplicate transition')
        require(record.duration >= 2, 'minimum hold')
        self._seen_digests.add(digest)
        self._seen_identities[key] = digest
        self._rows.append(record)
        self.accepted_count += 1
        while len(self._rows) > self.capacity:
            self._rows.popleft()
            self.evicted_count += 1

    def sample(self, batch_size, generator):
        require(isinstance(generator, torch.Generator) and generator is not torch.default_generator
                and generator.device.type == 'cpu', 'private CPU replay generator required')
        require(0 < batch_size <= len(self._rows), 'invalid replay batch size')
        ix = torch.randperm(len(self._rows), generator=generator, device='cpu')[:batch_size]
        rows = [self._rows[int(i)] for i in ix]
        values = {}
        for name in ('state', 'next_state', 'mode_id', 'q_e4', 'reward', 'duration', 'discount',
                     'has_next_state', 'bootstrap', 'terminated', 'truncated'):
            dtype = torch.float32 if name in ('state', 'next_state', 'reward', 'discount') else (
                torch.int64 if name in ('mode_id', 'q_e4', 'duration') else torch.bool)
            values['_' + name] = torch.tensor([getattr(r, name) if name != 'next_state' or
                r.next_state is not None else (0.0,) * WIDTH for r in rows], dtype=dtype)
        return RT.Run5ReplayBatchV1(**values, binding_sha256=self.binding_sha256)


class Trainer(RT.Run5HybridSacTrainerV1):
    """Inherited, unmodified SAC update_once; only construction/24-D guards differ."""
    def __init__(self, *, actor, critics, config, binding_sha256, target_generator, actor_generator):
        require(type(config) is RT.T.TrainerConfigV1, 'foreign config')
        validate_models(actor, critics)
        for g in (target_generator, actor_generator):
            require(isinstance(g, torch.Generator) and g is not torch.default_generator
                    and g.device.type == 'cpu', 'private CPU generator required')
        require(target_generator is not actor_generator, 'distinct generators required')
        self.actor, self.critics, self.config = actor, critics, config
        self.expected_binding = binding_sha256
        self._target_generator, self._actor_generator = target_generator, actor_generator
        self._online_critic_parameters = list(chain(critics.critic_1.parameters(), critics.critic_2.parameters()))
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self._online_critic_parameters, lr=config.critic_lr)
        self.update_count = 0
        self._assert_optimizer_wiring()

    def _preflight(self, batch):
        self._assert_optimizer_wiring()
        validate_models(self.actor, self.critics)
        require(type(batch) is RT.Run5ReplayBatchV1, 'foreign batch')
        self._require_evidence_class(batch.binding_sha256)
        n = batch.batch_size
        require(n > 0, 'empty batch')
        for name, t, shape in (('state', batch.state, (n, WIDTH)), ('next_state', batch.next_state, (n, WIDTH)),
                ('reward', batch.reward, (n,)), ('discount', batch.discount(), (n,)),
                ('q', batch.q_normalized_executed, (n,))):
            require(tuple(t.shape) == shape and t.dtype == torch.float32 and bool(torch.isfinite(t).all()),
                    'invalid ' + name)
        has, boot, term, trunc = batch.has_next_state, batch.bootstrap, batch.terminated, batch.truncated
        require(torch.equal(boot, has & ~term & ~trunc), 'invalid bootstrap mask')
        require(torch.equal(has, ~(term | trunc)), 'successor disagrees with boundary')
        require(not bool(batch.next_state[~has].abs().sum()), 'absent successor must be zero')
        require(bool(((batch.mode_id >= 0) & (batch.mode_id < 12)).all()), 'invalid mode')
        require(bool(((batch.q_e4 >= 0) & (batch.q_e4 <= 9800)).all()), 'invalid q')
        require(bool((batch.duration >= 2).all()), 'minimum hold')
        require(bool(((batch.discount() > 0) & (batch.discount() <= 1)).all()), 'invalid discount')
        # Encoder invariants on both stored states; vectorized to keep original update cost modest.
        for x in (batch.state, batch.next_state[has]):
            require(bool(((x[:, 22] == x[:, 20]) & (x[:, 20] <= x[:, 19])).all()), 'quality mask')
            require(bool((x[:, 4:16].sum(1) == x[:, 19]).all()), 'history mask')
            require(bool(((x[:, 23] >= 0) & (x[:, 23] <= 1)).all()), 'receipt range')
            absent_quality = x[:, 20] == 0
            require(not bool(x[absent_quality, 17:19].abs().sum()), 'fabricated quality/legacy latency')


class CandidateOrchestrator(TrialOrchestratorV1):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.collector = HistoryCollector(self.collector)
        self.actor, self.critics = extend_models(self.actor, self.critics)
        self.binding_document = dict(self.binding_document, candidate_schema=SCHEMA,
            candidate_collector=self.collector.collector_binding_sha256, state_dim=WIDTH,
            initialization='copy original seed17 initial function; two new zero columns')
        self.binding_sha256 = hashlib.sha256(json.dumps(self.binding_document, sort_keys=True).encode()).hexdigest()
        self.replay = Replay(RT.CONFIG.replay_capacity, self.binding_sha256)
        self.trainer = Trainer(actor=self.actor, critics=self.critics, config=RT.trainer_config(),
            binding_sha256=self.binding_sha256, target_generator=self.generators['trainer_target'],
            actor_generator=self.generators['trainer_actor'])

    @classmethod
    def restore(cls, *args, **kwargs):
        raise RuntimeError('No resume/retry is authorized for this one-pair study')
