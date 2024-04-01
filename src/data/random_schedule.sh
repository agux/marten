#!/bin/bash

# Get the directory where the schedule_random.sh script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Your actual script you want to run at a random time
ACTUAL_SCRIPT="run.sh"

# Calculate random hour and minute
HOUR=$((RANDOM % 4 + 17)) # 17 (5 PM) to 20 (8 PM), as the last job can start at 8:59 PM
MINUTE=$((RANDOM % 60))

# Schedule the task using at
at $HOUR:$MINUTE <<EOF
cd $SCRIPT_DIR
bash $ACTUAL_SCRIPT
EOF
