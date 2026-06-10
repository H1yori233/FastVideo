#!/usr/bin/env bash
# Deployment health check. Run on the node where you SERVE (the login node).
#   bash tools/check_deploy.sh [PORT]      (PORT default 4173 = vite preview)
APP="/home/hal-kaiqin/dreamverse/FastVideo_app/app"
PORT="${1:-4173}"
cd "$APP" 2>/dev/null || { echo "[FAIL] app dir missing: $APP"; exit 1; }

ok()   { printf '  \033[32m[OK]\033[0m   %s\n' "$1"; }
bad()  { printf '  \033[31m[FAIL]\033[0m %s\n' "$1"; }
warn() { printf '  \033[33m[WARN]\033[0m %s\n' "$1"; }
http() { curl -s -o /dev/null -w '%{http_code}' --max-time "${2:-5}" "$1" 2>/dev/null; }

echo "================ deploy check ($(date '+%H:%M:%S')) ================"
echo "host: $(hostname)   port: $PORT"

echo "-- 1. toolchain --"
if command -v node >/dev/null; then ok "node $(node -v)"
else warn 'node not on PATH -> export PATH="/home/hal-kaiqin/miniforge3/envs/FastVideo_kaiqin/bin:$PATH"'; fi

echo "-- 2. build artifacts (dist/) --"
if [ -f dist/index.html ]; then
  ok "dist/index.html"
  ls dist/assets/*.js >/dev/null 2>&1 && ok "JS bundle present" || bad "no dist/assets/*.js"
  [ -f dist/data.json ] && ok "dist/data.json ($(wc -c <dist/data.json)B)" || bad "dist/data.json missing"
  n=$(ls dist/thumbs 2>/dev/null | wc -l); [ "$n" -gt 0 ] && ok "dist/thumbs ($n jpg)" || warn "no thumbs"
  if [ -L dist/videos ]; then
    [ -e "dist/videos" ] && ok "dist/videos symlink resolves" || bad "dist/videos symlink BROKEN -> $(readlink dist/videos)"
  else bad "dist/videos symlink missing -> run: bash tools/build_static.sh"; fi
  if find src vite.config.ts svelte.config.js -type f -newer dist/index.html 2>/dev/null | grep -q .; then
    warn "source changed since last build -> rebuild: bash tools/build_static.sh"
  else ok "dist up to date with src"; fi
else
  bad "no dist/ -> run: bash tools/build_static.sh"
fi

echo "-- 3. preview server on :$PORT --"
code=$(http "http://localhost:$PORT/")
if [ "$code" = "200" ]; then
  ok "GET / -> 200"
  c=$(http "http://localhost:$PORT/data.json"); [ "$c" = 200 ] && ok "GET /data.json -> 200" || bad "GET /data.json -> $c"
  thumb=$(ls dist/thumbs 2>/dev/null | head -1)
  [ -n "$thumb" ] && { c=$(http "http://localhost:$PORT/thumbs/$thumb"); [ "$c" = 200 ] && ok "GET /thumbs/$thumb -> 200" || bad "poster -> $c"; }
  vid=$(python3 -c "import json;d=json.load(open('dist/data.json'));print(next(v['file_path'] for v in d['output_videos'] if v['exists']))" 2>/dev/null)
  [ -n "$vid" ] && { c=$(http "http://localhost:$PORT/$vid" 8); [ "$c" = 200 ] && ok "GET a real video -> 200" || bad "video -> $c (videos symlink?)"; }
else
  bad "GET / -> $code  (preview NOT running on :$PORT)"
  echo "        start it: bash tools/serve.sh $PORT"
fi

echo "-- 4. ngrok (public link) --"
if pgrep -x ngrok >/dev/null; then
  ok "ngrok process running"
  url=$(curl -s --max-time 5 http://localhost:4040/api/tunnels 2>/dev/null | grep -oE 'https://[a-z0-9.-]+\.ngrok[a-z.-]*' | head -1)
  [ -n "$url" ] && ok "PUBLIC URL: $url" || warn "ngrok up but no URL from :4040 (still connecting, or started elsewhere)"
else
  warn "no ngrok running (no public link). start: ~/bin/ngrok http $PORT"
fi

echo "-- 5. processes (sanity) --"
nv=$(pgrep -x node 2>/dev/null | while read -r p; do grep -qa 'bin/vite' /proc/$p/cmdline 2>/dev/null && echo x; done | wc -l)
echo "  vite/preview node procs: $nv"
[ "$nv" -gt 1 ] && warn "more than one vite running -> can corrupt node_modules/.vite. keep one."
echo "================ any [FAIL] = that link in the chain is broken ================"
