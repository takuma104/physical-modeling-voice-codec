#!/bin/bash
# The engine's Python module as a wheel for the Space's image (Linux x86_64, Python 3.13), built
# in that image: run on a Linux x86_64 machine with Docker, at the repository's root.
#   bash tools/wheel.sh        → dist/pmvc_engine-…-cp313-cp313-linux_x86_64.whl
set -e -o pipefail
mkdir -p dist
docker run --rm --cpus "${CPUS:-6}" -v "$PWD":/src -w /src -e OWNER="$(id -u):$(id -g)" python:3.13-slim bash -c '
  set -e
  apt-get update -qq && apt-get install -y -qq curl build-essential > /dev/null
  curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal > /dev/null
  . "$HOME/.cargo/env"
  pip install -q --root-user-action ignore maturin
  CARGO_TARGET_DIR=/tmp/target maturin build --release -m crates/pmvc-engine/Cargo.toml -o /src/dist --compatibility linux 2>&1 | tail -3
  chown -R "$OWNER" /src/dist'
ls -la dist/*.whl
