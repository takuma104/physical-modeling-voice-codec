#!/bin/bash
# The engine's Python module as a wheel for the Space (Linux x86_64, Python 3.12: what ZeroGPU
# runs), built in PyPA's manylinux image so that it loads with any glibc from 2.28 on: run on a
# Linux x86_64 machine with Docker, at the repository's root.
#   bash tools/wheel.sh        → dist/pmvc_engine-…-cp312-cp312-manylinux_2_28_x86_64.whl
#   PY=cp313 bash tools/wheel.sh    for another Python
set -e -o pipefail
mkdir -p dist
docker run --rm --cpus "${CPUS:-6}" -v "$PWD":/src -w /src -e OWNER="$(id -u):$(id -g)" -e PY="${PY:-cp312}" quay.io/pypa/manylinux_2_28_x86_64 bash -c '
  set -e
  curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal > /dev/null
  . "$HOME/.cargo/env"
  P=/opt/python/$PY-$PY/bin
  $P/pip install -q --root-user-action ignore maturin
  CARGO_TARGET_DIR=/tmp/target $P/maturin build --release -m crates/pmvc-engine/Cargo.toml -o /src/dist -i $P/python --compatibility manylinux_2_28 2>&1 | tail -3
  chown -R "$OWNER" /src/dist'
ls -la dist/*.whl
