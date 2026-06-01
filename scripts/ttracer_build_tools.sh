#!/usr/bin/env bash
# Build the small T-tracer tools needed for smoke recording/extraction.
set -euo pipefail
source "$(dirname "$0")/config.env"

TOOLS=(record replay csv extract_config textlog)

cd "${OAI_T_TRACER_DIR}"
echo "[ttracer_build_tools] building: ${TOOLS[*]}"
make "${TOOLS[@]}"

for tool in "${TOOLS[@]}"; do
  if [[ ! -x "${OAI_T_TRACER_DIR}/${tool}" ]]; then
    echo "[ttracer_build_tools] missing executable after build: ${tool}" >&2
    exit 1
  fi
done

echo "[ttracer_build_tools] ready in ${OAI_T_TRACER_DIR}"
