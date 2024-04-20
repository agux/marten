#!/bin/bash

# Base name of the Python program
BASE_SCRIPT_NAME="marten"

# Check if a parameter is provided
if [ $# -eq 0 ]; then
    echo "No additional keyword provided. Using default script name."
    SCRIPT_NAME=$BASE_SCRIPT_NAME
else
    # Combine the base script name with the provided parameter
    SCRIPT_NAME="$BASE_SCRIPT_NAME $1"
fi

# Use pgrep to find the PID of the script based on the combined name
MAIN_PID=$(pgrep -f "$SCRIPT_NAME")

# Check if the MAIN_PID variable is not empty
if [ ! -z "$MAIN_PID" ]; then
    echo "Killing main process and its children: $MAIN_PID"

    # Kill the main process and its child processes
    pkill -P $MAIN_PID
    kill $MAIN_PID

    echo "Processes killed."
else
    echo "No process found for $SCRIPT_NAME."
fi
