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

# Clear the logs
>etl.log
>$OUTPUT_LOG

# Run the script with nohup
# nohup "$PYENV_PYTHON" "$script" "$@" > >(tee -a output.log) 2>&1 &
nohup "$COMMAND" "etl" "--profile" "$@" > >(tee -a $OUTPUT_LOG) 2>&1 &