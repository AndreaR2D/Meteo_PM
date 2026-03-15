#!/bin/bash
# Setup script for VPS deployment.
# Run once after cloning the repo to configure the daily cron job.
#
# Usage:
#   chmod +x setup_cron.sh
#   ./setup_cron.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$(command -v python3 || command -v python)"

echo "=== Paper Trade Collector Setup ==="
echo "Script dir: $SCRIPT_DIR"
echo "Python:     $PYTHON"

# Install deps
echo "Installing dependencies..."
"$PYTHON" -m pip install -r "$SCRIPT_DIR/requirements.txt" --quiet

# Test run
echo "Running a test collection..."
cd "$SCRIPT_DIR" && "$PYTHON" collector.py
echo ""

# Setup cron: every day at 19h Paris time (CET=18h UTC, CEST=17h UTC)
# We schedule at 17:00 UTC to cover summer time (CEST).
# In winter (CET), this runs at 18h Paris = still fine.
CRON_CMD="0 17 * * * cd $SCRIPT_DIR && $PYTHON collector.py >> $SCRIPT_DIR/cron.log 2>&1"

# Check if cron entry already exists
if crontab -l 2>/dev/null | grep -q "collector.py"; then
    echo "Cron entry already exists. Skipping."
else
    (crontab -l 2>/dev/null; echo "$CRON_CMD") | crontab -
    echo "Cron job installed: $CRON_CMD"
fi

echo ""
echo "=== Done ==="
echo "The collector will run daily at 17:00 UTC (19h Paris summer / 18h Paris winter)."
echo "Logs:    $SCRIPT_DIR/collector.log"
echo "Data:    $SCRIPT_DIR/history.csv"
echo "Cron:    crontab -l"
