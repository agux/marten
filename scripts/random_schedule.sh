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
echo "cd $SCRIPT_DIR; bash $ACTUAL_SCRIPT >> $SCRIPT_DIR/random_schedule.log 2>&1" | at $TIME

# Schedule a second (backup) run 5 minutes later
BACKUP_MINUTE=$((MINUTE + 5))
if [ $BACKUP_MINUTE -ge 60 ]; then
  BACKUP_MINUTE=$((BACKUP_MINUTE - 60))
  BACKUP_HOUR=$((HOUR + 1))
else
  BACKUP_HOUR=$HOUR
fi

# Format the backup time for the at command
BACKUP_TIME=$(printf "%02d:%02d" $BACKUP_HOUR $BACKUP_MINUTE)

# Schedule the backup task using at
echo "cd $SCRIPT_DIR; bash $ACTUAL_SCRIPT >> $SCRIPT_DIR/random_schedule.log 2>&1" | at $BACKUP_TIME
