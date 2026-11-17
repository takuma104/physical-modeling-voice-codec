#!/bin/bash
# The Hugging Face Space, assembled from this repository into a clone of the Space's own.
#   bash tools/space.sh ../pmvc-tts-space [dist/pmvc_engine-….whl]
#   MODELS=../pmvc-tts bash tools/space.sh ../pmvc-tts-space      with the models in the Space itself
# Needs the page's engine built (cargo build -p pmvc-wasm --release --target wasm32-unknown-unknown)
# and the wheel (tools/wheel.sh, on a Linux x86_64 machine). Everything the Space has but its
# .git and .gitattributes is replaced. The wheel is unpacked into engine/: a Gradio Space
# installs what requirements.txt names before it has the repository's files, so it cannot
# install a wheel that is among them, and app.py finds the module there instead. With MODELS,
# that directory's published models (<lang>.json and <lang>.safetensors, `pmvc.tts export`) go
# into the Space's models/, and the Space then asks nothing of the model repository when it starts.
set -e -o pipefail
D=${1:?where the clone of the Space is}
WHEEL=${2:-$(ls dist/pmvc_engine-*-manylinux*_x86_64.whl | tail -1)}
WASM=target/wasm32-unknown-unknown/release/pmvc_wasm.wasm
C=third-party/chimera-vocal-synth
test -d "$D/.git" && test -f "$WHEEL" && test -f "$WASM"
find "$D" -mindepth 1 -maxdepth 1 ! -name .git ! -name .gitattributes -exec rm -rf {} +
mkdir -p "$D/pmvc" "$D/voices" "$D/engine" "$D/$C/public/data"
cp space/README.md space/app.py space/requirements.txt LICENSE "$D/"
# the modules the server loads, and no others
for m in $(uv run python -c 'import sys, pmvc.serve, pmvc.english; print(" ".join(sorted(m.split(".")[1] for m in sys.modules if m.startswith("pmvc."))))'); do
  cp "pmvc/$m.py" "$D/pmvc/"
done
cp pmvc/__init__.py "$D/pmvc/"
cp -R web "$D/web"
cp voices/v1.json "$D/voices/"
cp "$WASM" "$D/pmvc_wasm.wasm"
unzip -q "$WHEEL" 'pmvc_engine/*' -d "$D/engine"
cp "$C/LICENSE" "$D/$C/LICENSE"
cp "$C/public/data/cmudict.txt.gz" "$C/public/data/CMUDICT-LICENSE.txt" "$D/$C/public/data/"
if [ -n "$MODELS" ]; then
  mkdir -p "$D/models"
  for j in "$MODELS"/*.json; do
    test -f "${j%.json}.safetensors" && cp "$j" "${j%.json}.safetensors" "$D/models/"
  done
fi
git -C "$C" rev-parse HEAD > "$D/$C/COMMIT"
git rev-parse HEAD > "$D/COMMIT"
grep -q '\*.so' "$D/.gitattributes" || echo '*.so filter=lfs diff=lfs merge=lfs -text' >> "$D/.gitattributes"
du -sh "$D" --exclude .git 2>/dev/null || du -sh "$D"
ls "$D" "$D/pmvc"
