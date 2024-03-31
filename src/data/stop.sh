#!/bin/bash

# Name of the Python script without the .py extension
SCRIPT_NAME="etl.py"

# Use pgrep to find the PID of the main script based on its name
MAIN_PID=$(pgrep -f $SCRIPT_NAME)

if [ ! -z "$MAIN_PID" ]; then
    echo "Killing main process and its children: $MAIN_PID"

    # Kill the main process and its child processes
    pkill -P $MAIN_PID
    kill $MAIN_PID

    echo "Processes killed."
else
    echo "No process found for $SCRIPT_NAME."
fi
