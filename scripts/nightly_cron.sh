#!/bin/bash
# Nightly Optimization for Project Laplace
# Suggested Run Time: 16:30 IST (after market close and data settlement)

# Navigate to project root (one level above scripts/)
cd "$(dirname "$0")/.."

# ── Load environment variables (5Paisa creds, USE_LIVE_DATA, etc.) ──────────
# Without this, FivePaisaHelper sees no credentials and aborts refinement.
if [ -f config/.env ]; then
    set -a
    # shellcheck disable=SC1091
    source config/.env
    set +a
else
    echo "[$(date)] WARNING: config/.env not found. Refinement may fail." >> logs/nightly_ops.log
fi

export PYTHONPATH="$PYTHONPATH:."

echo "[$(date)] Starting Nightly Refinement..." >> logs/nightly_ops.log
python3 -u rl_trading/refine_daily.py >> logs/nightly_ops.log 2>&1
echo "[$(date)] Refinement Complete." >> logs/nightly_ops.log
