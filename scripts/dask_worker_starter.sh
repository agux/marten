#!/bin/bash

SCRIPT="dask_workers.sh"

# Check if 'dask worker' process is already running
if pgrep -af 'dask worker' | grep -v 'grep'; then
  echo "The process 'dask worker' is already running."
  exit 1
fi

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd $SCRIPT_DIR

# Source the .env file from the parent directory
if [ -f ../.env ]; then
  source ../.env
else
  echo ".env file not found in the parent directory!"
  exit 1
fi

# Check if the remote port is connectable using nc
if ! nc -z -w5 $DASK_SCHEDULER_IP $DASK_SCHEDULER_PORT; then
  echo "Error: Cannot connect to $DASK_SCHEDULER_IP on port $DASK_SCHEDULER_PORT"
  exit 1
fi

# Load pyenv into the shell session.
# export PYENV_ROOT="$HOME/.pyenv"
# export PATH="$PYENV_ROOT/bin:$PATH"
# if command -v pyenv 1>/dev/null 2>&1; then
#   eval "$(pyenv init -)"
# fi

bash $SCRIPT $DASK_SCHEDULER_IP $DASK_SCHEDULER_PORT $DASK_WORKERS $DASK_THREADS $DASK_POWER
