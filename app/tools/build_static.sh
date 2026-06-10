#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT="$(readlink public/videos 2>/dev/null || true)"
if [ -z "$CKPT" ]; then
  echo "public/videos symlink missing — run: python3 tools/gen_demo_data.py" >&2
  exit 1
fi

rm -f public/videos
npm run build
ln -sfn "$CKPT" public/videos
ln -sfn "$CKPT" dist/videos

echo "built dist/ — videos symlinked (not copied) -> $CKPT"
