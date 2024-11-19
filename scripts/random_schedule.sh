#!/bin/bash

# Get the directory where the schedule_random.sh script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Your actual script you want to run at a random time
ACTUAL_SCRIPT="run_etl.sh"

# Calculate random hour and minute, 18:00 - 18:59
HOUR=18
MINUTE=$((RANDOM % 60))

# Format the time for the at command
TIME=$(printf "%02d:%02d" $HOUR $MINUTE)

# Schedule the task using at
at $TIME <<EOT
cd $SCRIPT_DIR
if [ \$(date +\%u) -eq 6 ]; then
    bash $ACTUAL_SCRIPT >> $SCRIPT_DIR/random_schedule.log 2>&1
else
    bash $ACTUAL_SCRIPT --exclude=calc_ta >> $SCRIPT_DIR/random_schedule.log 2>&1
fi
EOT
