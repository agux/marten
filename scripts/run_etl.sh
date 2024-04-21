#!/bin/bash

# Load pyenv into the shell session.
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
if command -v pyenv 1>/dev/null 2>&1; then
  eval "$(pyenv init -)"
fi

COMMAND="$(pyenv which marten)"
OUTPUT_LOG=output_etl.log
# script=etl.py
# PYENV_PYTHON=$(pyenv which python)

# Check if 'marten etl' process is already running
if pgrep -af 'marten etl' | grep -v 'grep'; then
  echo "The process 'marten etl' is already running."
  exit 1
fi

# Clear the logs
>marten.data.etl.log
>marten.cli.commands.etl.log
>etl.log
>$OUTPUT_LOG

# Run the script with nohup
# nohup "$PYENV_PYTHON" "$script" "$@" > >(tee -a output.log) 2>&1 &
nohup "$COMMAND" "etl" "$@" > >(tee -a $OUTPUT_LOG) 2>&1 &
