#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd $SCRIPT_DIR

if [ -f ../.env ]; then
  source ../.env
fi

OUTPUT_LOG=output_etl.log
PID_FILE=etl_process.pid

# Read the PID from the file
if [ ! -f $PID_FILE ]; then
  echo "PID file not found!" >> $OUTPUT_LOG
  exit 1
fi
PID=$(cat $PID_FILE)

# Wait for the process to complete
while kill -0 $PID 2>/dev/null; do
  sleep 1
done

# Send email notification with the output log
cat $OUTPUT_LOG | mail -s "$EMAIL_SUBJECT" $EMAIL_ADDR

# Clean up the PID file
rm -f $PID_FILE