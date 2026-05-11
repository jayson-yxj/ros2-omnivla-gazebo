#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
THIRD_PARTY="${PROJECT_ROOT}/third_party"
GITHUB_PREFIX="${GITHUB_PREFIX:-}"

repo_url() {
  local url="$1"
  if [[ -n "${GITHUB_PREFIX}" ]]; then
    printf '%s%s\n' "${GITHUB_PREFIX}" "${url}"
  else
    printf '%s\n' "${url}"
  fi
}

clone_if_missing() {
  local url="$1"
  local dir="$2"
  if [[ -d "${dir}/.git" ]]; then
    printf '[skip] %s already exists\n' "${dir}"
    return
  fi
  git clone --depth 1 "$(repo_url "${url}")" "${dir}"
}

patch_omnivla_imports() {
  local root="${THIRD_PARTY}/OmniVLA"
  if [[ ! -d "${root}" ]]; then
    printf '[error] OmniVLA repo not found at %s\n' "${root}" >&2
    return 1
  fi

  cat > "${root}/prismatic/__init__.py" <<'PY'
"""Prismatic package.

Keep package import side-effect free for inference users.
The original project eagerly imports training/data modules here, which drags in
TensorFlow/RLDS dependencies even when only the inference path is needed.
"""
PY

  cat > "${root}/prismatic/models/__init__.py" <<'PY'
"""Model registry package.

Avoid eager import of training/materialization helpers at package import time.
Inference code can import concrete submodules directly.
"""
PY

  cat > "${root}/prismatic/vla/__init__.py" <<'PY'
"""VLA package.

Keep import lightweight for inference-only users.
"""
PY

  cat > "${root}/prismatic/training/__init__.py" <<'PY'
"""Training package.

Do not import training materialization automatically.
"""
PY
}

mkdir -p "${THIRD_PARTY}"
touch "${THIRD_PARTY}/COLCON_IGNORE"

clone_if_missing "https://github.com/NHirose/OmniVLA.git" "${THIRD_PARTY}/OmniVLA"
clone_if_missing "https://github.com/aws-robotics/aws-robomaker-small-house-world.git" "${THIRD_PARTY}/aws-robomaker-small-house-world"

patch_omnivla_imports

printf '[done] third-party repositories are ready under %s\n' "${THIRD_PARTY}"
