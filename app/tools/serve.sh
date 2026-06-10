#!/usr/bin/env bash
# Clean (re)start of the static preview server. Run on the node where you SERVE.
#   bash tools/serve.sh [PORT]      (PORT default 4173)
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="/home/hal-kaiqin/miniforge3/envs/FastVideo_kaiqin/bin:$PATH"
PORT="${1:-4173}"

echo "[serve] stopping any running vite/preview ..."
pkill -f 'node_modules/.bin/vite' 2>/dev/null || true
sleep 1

echo "[serve] building static dist/ ..."
bash tools/build_static.sh

echo "[serve] starting preview on :$PORT ..."
nohup npm run preview -- --port "$PORT" >/tmp/preview.log 2>&1 &
sleep 4
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://localhost:$PORT/" || echo 000)
echo "[serve] preview on :$PORT -> HTTP $code   (log: /tmp/preview.log)"
if [ "$code" = "200" ]; then
  echo "[serve] OK. To expose publicly:  ~/bin/ngrok http $PORT"
  echo "[serve] Or open via VS Code Ports -> forward $PORT -> localhost:$PORT"
else
  echo "[serve] preview did not answer 200 — check /tmp/preview.log"
fi
