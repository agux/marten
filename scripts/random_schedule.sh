#!/bin/bash

# Get the directory where the schedule_random.sh script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Your actual script you want to run at a random time
ACTUAL_SCRIPT="run_etl.sh"

# Calculate random hour and minute, 16:15 - 16:59
HOUR=16
MINUTE=$((RANDOM % 45 + 15))

# Schedule the task using at
at $HOUR:$MINUTE <<EOF
cd $SCRIPT_DIR
bash $ACTUAL_SCRIPT
EOF
