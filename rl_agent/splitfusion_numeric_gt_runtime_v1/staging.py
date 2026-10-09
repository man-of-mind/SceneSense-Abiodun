"""Versioned source staging before the policy chooses its action.

Only the two action-dependent capture fields are left empty. The native
sensor/world snapshot is frozen before action-open, then the exact selected
capture identity is bound by a small message. No GT calculation runs on the
template, and no template can be published as a scoring target.
"""
from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from .source_codec import encode_source, decode_source

SCHEMA = "splitfusion.numeric_gt.early_source_stage.v1"
_ACTION_FIELDS = ("action_id", "profile_id")


def neutral_identity(capture):
    original = C._json_owned(capture)
    C._capture(original)
    return dict(original, action_id=None, profile_id=None)


def encode_stage(sourcekwargs, policy_timestamp_s):
    """Freeze exact source bytes outside action-open, with no chosen action."""
    C.require(type(sourcekwargs) is dict and "identity" in sourcekwargs,
              "owned source fields required")
    original = C._json_owned(sourcekwargs["identity"])
    C._capture(original)
    template = neutral_identity(original)
    chunks, extras = encode_source(dict(sourcekwargs, identity=template), policy_timestamp_s)
    C.require(extras["capture_identity"] == template, "staged capture template drift")
    return chunks, dict(schema=SCHEMA, source_extras=extras,
                        capture_template_sha256=C.sha256(C.canonical(template)))


def decode_stage(packet, stage_extras):
    C.require(type(stage_extras) is dict and set(stage_extras) ==
              {"schema", "source_extras", "capture_template_sha256"}
              and stage_extras["schema"] == SCHEMA, "early source staging schema differs")
    source = decode_source(packet, stage_extras["source_extras"])
    template = source["identity"]
    C.require(template == neutral_identity(template) and C.sha256(C.canonical(template))
              == stage_extras["capture_template_sha256"], "staged source contains an action or wrong template")
    return source


def bind_action(source, capture_identity):
    """Rebind only action/profile once the selected full plan is known."""
    C.require(type(source) is dict and "identity" in source, "decoded staged source required")
    template = C._json_owned(source["identity"])
    C._capture(template)
    C.require(template == neutral_identity(template), "staged source is already action-bound")
    selected = C._json_owned(capture_identity)
    C._capture(selected)
    C.require(neutral_identity(selected) == template,
              "planned capture differs from the owned staged sensor frame")
    return dict(source, identity=selected)
