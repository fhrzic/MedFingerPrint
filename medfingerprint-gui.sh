#!/usr/bin/env bash
# Start the medfingerprint GUI (double-click or run from a terminal).
# Uses the virtual environment next to this script (.venv), or "mfp" on the PATH.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -x "$DIR/.venv/bin/mfp" ]; then
    exec "$DIR/.venv/bin/mfp" gui "$@"
fi
exec mfp gui "$@"
