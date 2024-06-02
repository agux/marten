#!/bin/bash

# Get the number of CPU cores
NUM_CORES=$(nproc)

# Calculate 90% of the number of CPU cores
DEFAULT_NPROCS=$(echo "$NUM_CORES * 0.9 / 1" | bc)

# Default values
DEFAULT_SCHEDULER_IP="127.0.0.1"
DEFAULT_PORT="8786"
DEFAULT_NTHREADS="1"

# Parameters with default values
SCHEDULER_IP=${1:-$DEFAULT_SCHEDULER_IP}
PORT=${2:-$DEFAULT_PORT}
NPROCS=${3:-$DEFAULT_NPROCS}
NTHREADS=${4:-$DEFAULT_NTHREADS}

OUTPUT_LOG=workers.log

# Clear the log file
> $OUTPUT_LOG

# Start the Dask worker with the specified parameters
nohup dask-worker tcp://$SCHEDULER_IP:$PORT --nprocs $NPROCS --nthreads $NTHREADS > >(tee -a $OUTPUT_LOG) 2>&1 &
