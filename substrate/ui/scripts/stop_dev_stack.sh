#!/usr/bin/env bash
set -euo pipefail

UI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

stop_pid() {
  local pid="$1"
  local reason="$2"
  if [[ -z "$pid" ]] || [[ ! -d "/proc/$pid" ]]; then
    return 0
  fi
  local cwd
  cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
  local cmd
  cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || ps -p "$pid" -o args= 2>/dev/null || true)"
  if [[ "$cwd" == "$UI_DIR"* ]]; then
    echo "stopping pid=$pid reason=$reason cmd=$cmd"
    kill "$pid" 2>/dev/null || true
  fi
}

for port in 5173 8787; do
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    stop_pid "$pid" "port:$port"
  done < <(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null | sort -u)
done

sleep 1

while read -r pid args; do
  [[ -n "$pid" ]] || continue
  case "$args" in
    *"/Documents/MNIST/ui/node_modules/.bin/concurrently"*|*"/Documents/MNIST/ui/node_modules/.bin/vite"*|"sh -c vite"|"sh -c node api/server.mjs"|"node api/server.mjs")
      stop_pid "$pid" "wrapper"
      ;;
  esac
done < <(ps -eo pid=,args=)
