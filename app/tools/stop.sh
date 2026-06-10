#!/usr/bin/env bash
pkill -f 'node_modules/.bin/vite' && echo "[stop] vite preview stopped" || echo "[stop] no vite preview running"
pkill -x ngrok 2>/dev/null && echo "[stop] ngrok stopped" || true
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://localhost:${1:-4173}/" 2>/dev/null || echo 000)
echo "[stop] port ${1:-4173} -> HTTP $code (000 = down)"
