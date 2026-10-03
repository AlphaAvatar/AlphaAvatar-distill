#!/usr/bin/env bash
# Behavioural test for the cold-host tripwire, extracted from e5_setup.sh and
# driven against synthetic hosts. It is a shell test because the thing under
# test is shell: a python reimplementation would prove nothing about the code
# that actually runs on a pod.
#
# Three hosts are simulated:
#   warm     -- uv sync finishes fast                      -> proceed
#   cold     -- uv sync never finishes, nothing grows       -> trip, exit 90
#   linking  -- uv sync slow but site-packages growing fast -> one grace, proceed
#
# Run: bash tests/pod/test_cold_host_tripwire.sh
set -uo pipefail
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0
check() { if [ "$2" = "$3" ]; then echo "  PASS $1"; pass=$((pass+1));
          else echo "  FAIL $1: got '$2' want '$3'"; fail=$((fail+1)); fi }

# The tripwire, lifted verbatim in structure from e5_setup.sh. VENV and the
# sync command are injected so a test can shape the host's behaviour.
tripwire() {
  local TRIP_S=$1 GRACE_S=$2 VENV=$3; shift 3
  say() { :; }; mark() { echo "MARK:$1" >> "$TMP/marks"; }
  "$@" & local UV_PID=$!
  local t0 el graced a b
  t0=$(date -u +%s); graced=0
  while kill -0 "$UV_PID" 2>/dev/null; do
    sleep 1
    el=$(( $(date -u +%s) - t0 ))
    [ "$el" -lt "$TRIP_S" ] && continue
    if [ "$graced" -eq 0 ] && [ -x "$VENV/bin/python" ]; then
      a=$(du -sb "$VENV" 2>/dev/null | cut -f1 || echo 0); sleep 2
      b=$(du -sb "$VENV" 2>/dev/null | cut -f1 || echo 0)
      if [ "$((b - a))" -gt 2000000 ]; then
        graced=1; TRIP_S=$(( TRIP_S + GRACE_S )); continue
      fi
    fi
    kill -9 "$UV_PID" 2>/dev/null || true
    mark "HOST_COLD:${el}s"
    return 90
  done
  wait "$UV_PID" || return 1
  return 0
}

echo "cold-host tripwire"

# --- warm: finishes well inside the window ---------------------------------
mkdir -p "$TMP/warm"
tripwire 5 3 "$TMP/warm" sleep 1
check "warm host proceeds" "$?" "0"

# --- cold: never finishes, nothing on disk ---------------------------------
rm -f "$TMP/marks"; mkdir -p "$TMP/cold"
tripwire 3 2 "$TMP/cold" sleep 300
check "cold host trips" "$?" "90"
check "cold host is marked" "$(grep -c HOST_COLD "$TMP/marks" 2>/dev/null)" "1"

# --- linking: past the window, but the venv is growing fast ----------------
# A host that is genuinely finishing must not be thrown away. Growth is faked
# by a writer appending to the venv while the "sync" runs on.
rm -f "$TMP/marks"; mkdir -p "$TMP/link/bin"
printf '#!/bin/sh\n' > "$TMP/link/bin/python"; chmod +x "$TMP/link/bin/python"
( for i in $(seq 1 12); do
    dd if=/dev/zero of="$TMP/link/blob.$i" bs=1M count=3 2>/dev/null; sleep 1
  done ) &
GROWER=$!
tripwire 3 8 "$TMP/link" sleep 7
rc=$?
kill -9 "$GROWER" 2>/dev/null
check "linking host is granted grace, not killed" "$rc" "0"
check "linking host is NOT marked cold" "$(grep -c HOST_COLD "$TMP/marks" 2>/dev/null || echo 0)" "0"

# --- the grace is bounded: one extension only, then it trips ---------------
rm -f "$TMP/marks"; mkdir -p "$TMP/slow/bin"
printf '#!/bin/sh\n' > "$TMP/slow/bin/python"; chmod +x "$TMP/slow/bin/python"
( for i in $(seq 1 3); do
    dd if=/dev/zero of="$TMP/slow/blob.$i" bs=1M count=3 2>/dev/null; sleep 1
  done ) &
GROWER=$!
tripwire 3 4 "$TMP/slow" sleep 300
check "grace is granted once, then the host trips" "$?" "90"
kill -9 "$GROWER" 2>/dev/null

# --- a real sync failure is NOT a cold host --------------------------------
rm -f "$TMP/marks"; mkdir -p "$TMP/broken"
tripwire 20 5 "$TMP/broken" false
check "genuine sync failure returns 1, not 90" "$?" "1"

echo "  $pass passed, $fail failed"
[ "$fail" -eq 0 ]
