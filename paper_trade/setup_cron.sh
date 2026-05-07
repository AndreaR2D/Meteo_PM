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

# Setup cron: daily at 07:00 UTC (09h Paris summer / 08h Paris winter)
# This ensures both ECMWF 00z (~06-07h UTC) and GFS 00z (~03:30-04:30h UTC)
# are available on Open-Meteo before we fetch.
CRON_CMD="0 7 * * * cd $SCRIPT_DIR && $PYTHON collector.py >> $SCRIPT_DIR/cron.log 2>&1"

# Check if cron entry already exists
if crontab -l 2>/dev/null | grep -q "collector.py"; then
    echo "Cron entry already exists. Skipping."
else
    (crontab -l 2>/dev/null; echo "$CRON_CMD") | crontab -
    echo "Cron job installed: $CRON_CMD"
fi

echo ""
echo "=== Done ==="
echo "The collector will run daily at 07:00 UTC (09h Paris summer / 08h Paris winter)."
echo "Each run: J-2, J-1, J forecasts for today + PM resolution for yesterday."
echo "Logs:    $SCRIPT_DIR/collector.log"
echo "Data:    $SCRIPT_DIR/history.csv"
echo "Cron:    crontab -l"
