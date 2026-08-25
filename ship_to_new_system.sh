#!/usr/bin/env bash
# Ship abiodun/ (code + docs + configs + checkpoints + OAI source) and the Claude memory to a new
# system. NON-DESTRUCTIVE: no --delete, nothing removed locally. DRY-RUN by default.
#
#   DRY_RUN=1 (default)  -> rsync -n, shows what WOULD transfer, mutates NEITHER machine
#   DRY_RUN=0            -> actually transfers
#
# Usage:
#   bash ship_to_new_system.sh              # dry run
#   DRY_RUN=0 bash ship_to_new_system.sh    # real transfer
#
# What ships:
#   - all abiodun/ code, docs, configs, study summaries, .git (full history)
#   - the dirty OAI source tree (compiled build dir excluded — must be rebuilt anyway)
#   - experiments/ae_integrated_20260710/ ONLY (noAE / AE32 / AE64 / AE128 checkpoints,
#     splits, summaries) — the rest of experiments/ is legacy and deliberately left behind
#   - the Claude project memory + user settings.json (permissions/model only)
# What never ships: credentials/tokens (~/.claude/.credentials.json), ~/.codex, bulk capture
# and experiment output trees (see .rsync-transfer-excludes).
set -euo pipefail

DEST_HOST="${DEST_HOST:-shr_aisvcs@W10275.idcc.lab}"
DEST_SHORTNAME="${DEST_HOST##*@}"; DEST_SHORTNAME="${DEST_SHORTNAME%%.*}"
# Same absolute paths on both machines (new box mirrors the local structure).
ABI="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"
DEST_PARENT="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab"
CLAUDE_PROJ="/home/shr_aisvcs/.claude/projects/-home-shr-aisvcs-workarea-carla-0-10-env-Carla-0-10-0-Linux-Shipping-PythonAPI-neu-collab"

# The single legacy experiment tree we still need on the new box.
AE_TREE="experiments/ae_integrated_20260710"

DRY_RUN="${DRY_RUN:-1}"
NFLAG=""; [ "$DRY_RUN" = "1" ] && NFLAG="-n" && echo "=== DRY RUN (set DRY_RUN=0 to transfer for real) ==="

# -a archive, -H hardlinks, -z compress, --partial resume-safe, progress2 overall bar. NO --delete.
RSYNC="rsync -aHz --partial --info=progress2 $NFLAG"

# Remote mkdir that is a NO-OP during a dry run (a dry run must not mutate either machine).
remote_mkdir() {
  if [ "$DRY_RUN" = "1" ]; then
    echo "    (dry run: would mkdir -p '$1' on ${DEST_HOST})"
  else
    ssh "$DEST_HOST" "mkdir -p '$1'" 2>/dev/null || true
  fi
}

echo ">>> [1/4] abiodun code+docs+configs+checkpoints+OAI-source -> ${DEST_HOST}"
echo "    (excludes: $ABI/.rsync-transfer-excludes)"
# NOTE: source needs a TRAILING SLASH so the leading-slash excludes anchor to abiodun/ (fixed 2026-08-04).
# Without it, the transfer root is the parent and `/metrics_logs/` etc. never match -> 654 GB of junk ships.
$RSYNC --exclude-from="$ABI/.rsync-transfer-excludes" \
  "$ABI/" "${DEST_HOST}:${DEST_PARENT}/abiodun/"

echo ">>> [2/4] ${AE_TREE} (noAE / AE32 / AE64 / AE128 checkpoints, splits, summaries)"
echo "    (the other legacy experiments/ trees are intentionally NOT transferred)"
remote_mkdir "${DEST_PARENT}/abiodun/experiments"
# No trailing slash on the source: recreates .../experiments/ae_integrated_20260710/ on the far side.
$RSYNC "$ABI/${AE_TREE}" "${DEST_HOST}:${DEST_PARENT}/abiodun/experiments/"

echo ">>> [3/4] Claude project memory (durable session context) -> same path on new box"
echo "    (memory only — no .credentials.json, no tokens, no .codex)"
remote_mkdir "$CLAUDE_PROJ"
$RSYNC "$CLAUDE_PROJ/memory" "${DEST_HOST}:${CLAUDE_PROJ}/"

echo ">>> [4/4] user Claude settings (permissions/model only) -> new box"
$RSYNC "/home/shr_aisvcs/.claude/settings.json" "${DEST_HOST}:/home/shr_aisvcs/.claude/" 2>/dev/null || \
  echo "    (settings.json skipped)"

cat <<NEXT

=== after the transfer completes, on ${DEST_SHORTNAME} ===
 1. Rebuild OAI (required — T-tracer byte-compares T_messages.txt vs the compiled copy):
      cd .../abiodun/OAI/openairinterface5g/cmake_targets && ./build_oai ... (gNB + nr-UE)
 2. Re-clone V2Xverse if needed:  git clone https://github.com/CollaborativePerception/V2Xverse
 3. Smoke-check: \`git -C .../abiodun status\` (should match), run a small offline script.
 4. Optional — the offline eval dataset was NOT shipped (it is 75k symlinks -> 90 GB of raw).
    If you need to reproduce offline evals, materialize the merged set (deref symlinks):
      rsync -aL <local>:.../fusion_training_data/moving_ego_pps200000_merged_8loops_stride2 \\
               .../fusion_training_data/
    or regenerate via the CARLA collector.
 5. Start Claude from the SAME directory you use locally so the memory hash matches and context auto-loads.

 !! NESTED OAI REPO: OAI/openairinterface5g/ is its own git repository with a DIRTY working tree.
    A push/pull of the root abiodun repo will NOT synchronize those uncommitted changes — this
    rsync is the only thing carrying them to ${DEST_SHORTNAME}. Re-verify after transfer with:
      git -C .../abiodun/OAI/openairinterface5g status
NEXT
