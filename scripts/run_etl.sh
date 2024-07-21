#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd $SCRIPT_DIR

if [ -f ../.env ]; then
  source ../.env
fi

# Load pyenv into the shell session.
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
if command -v pyenv 1>/dev/null 2>&1; then
  eval "$(pyenv init -)"
fi

COMMAND="$(pyenv which marten)"
OUTPUT_LOG=output_etl.log
PID_FILE=etl_process.pid

# Check if 'marten etl' process is already running
if pgrep -af 'marten etl' | grep -v 'grep'; then
  echo "The process 'marten etl' is already running." >> $OUTPUT_LOG
  exit 1
fi

# Clear the logs
>marten.data.etl.log
>marten.cli.commands.etl.log
>etl.log
>$OUTPUT_LOG

# Archive .csv files
if [ ! -d "csv" ]; then
  mkdir csv
fi
# Check if there are any .csv files in the current directory
csv_files=$(find . -maxdepth 1 -name "*.csv" -print)
if [ -n "$csv_files" ]; then
  # If .csv files are found, move them to the csv directory
  mv *.csv csv/
fi

export MALLOC_TRIM_THRESHOLD_=0
# Run the script with nohup
nohup "$COMMAND" "etl" "$@" > >(tee -a $OUTPUT_LOG) 2>&1 &
PID=$!

# Write the PID to a file
echo $PID > $PID_FILE

# Schedule the monitoring script to run in 1 minute
at now + 1 minute <<EOF
bash $SCRIPT_DIR/monitor_etl.sh
EOF