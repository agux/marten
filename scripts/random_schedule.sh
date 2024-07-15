#!/bin/bash

# Get the directory where the schedule_random.sh script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Your actual script you want to run at a random time
ACTUAL_SCRIPT="run_etl.sh"

# Calculate random hour and minute, 18:00 - 18:59
HOUR=18
MINUTE=$((RANDOM % 60))

# Schedule the task using at
at $HOUR:$MINUTE <<EOF
cd $SCRIPT_DIR
bash $ACTUAL_SCRIPT
EOF

# Schedule a second (backup) run 5 minutes later
BACKUP_MINUTE=$((MINUTE + 5))
if [ $BACKUP_MINUTE -ge 60 ]; then
  BACKUP_MINUTE=$((BACKUP_MINUTE - 60))
  BACKUP_HOUR=$((HOUR + 1))
else
  BACKUP_HOUR=$HOUR
fi

at $BACKUP_HOUR:$BACKUP_MINUTE <<EOF
cd $SCRIPT_DIR
bash $ACTUAL_SCRIPT
EOF