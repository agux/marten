#!/bin/bash

# Get the number of CPU cores
NUM_CORES=$(nproc)

# Calculate 90% of the number of CPU cores
DEFAULT_NWORKERS=$(echo "$NUM_CORES * 0.9 / 1" | bc)

# Default values
DEFAULT_SCHEDULER_IP="127.0.0.1"
DEFAULT_PORT="8786"
DEFAULT_NTHREADS="1"

# Parameters with default values
SCHEDULER_IP=${1:-$DEFAULT_SCHEDULER_IP}
PORT=${2:-$DEFAULT_PORT}
NWORKERS=${3:-$DEFAULT_NWORKERS}
NTHREADS=${4:-$DEFAULT_NTHREADS}

OUTPUT_LOG=workers.log

# Clear the log file
> $OUTPUT_LOG

# Clear old lr finder .ckpt files
rm -f .lr_find_*.ckpt

# Load pyenv into the shell session.
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
if command -v pyenv 1>/dev/null 2>&1; then
  eval "$(pyenv init -)"
fi

export MALLOC_TRIM_THRESHOLD_=0
# Start the Dask worker with the specified parameters
nohup dask worker tcp://$SCHEDULER_IP:$PORT --nworkers $NWORKERS --nthreads $NTHREADS --memory-limit 0 > >(tee -a $OUTPUT_LOG) 2>&1 &
