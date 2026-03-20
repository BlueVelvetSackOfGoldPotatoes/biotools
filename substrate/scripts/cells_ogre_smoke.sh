#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIDEcar_DIR="$ROOT_DIR/models/cells/standalone_ogre"
REQUIRE_OGRE="${REQUIRE_OGRE:-0}"

if ! command -v cmake >/dev/null 2>&1; then
  echo "[ogre-smoke] SKIP: cmake not found"
  exit 0
fi

if [[ ! -f "$SIDEcar_DIR/CMakeLists.txt" ]]; then
  echo "[ogre-smoke] FAIL: sidecar CMakeLists not found"
  exit 1
fi

BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/cells-ogre-smoke.XXXXXX")"
trap 'rm -rf "$BUILD_DIR"' EXIT

CONFIG_LOG="$BUILD_DIR/configure.log"
BUILD_LOG="$BUILD_DIR/build.log"

if ! cmake -S "$SIDEcar_DIR" -B "$BUILD_DIR" >"$CONFIG_LOG" 2>&1; then
  if grep -qi "Could not find a package configuration file provided by \"OGRE\"" "$CONFIG_LOG"; then
    if [[ "$REQUIRE_OGRE" == "1" ]]; then
      cat "$CONFIG_LOG"
      echo "[ogre-smoke] FAIL: OGRE is required but not available"
      exit 1
    fi
    echo "[ogre-smoke] SKIP: OGRE package not available"
    exit 0
  fi
  cat "$CONFIG_LOG"
  echo "[ogre-smoke] FAIL: configure error"
  exit 1
fi

if ! cmake --build "$BUILD_DIR" -j >"$BUILD_LOG" 2>&1; then
  cat "$BUILD_LOG"
  echo "[ogre-smoke] FAIL: build error"
  exit 1
fi

echo "[ogre-smoke] PASS: configured and built standalone OGRE sidecar"
