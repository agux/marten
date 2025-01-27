#!/bin/bash

ENV_NAME="python312"

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd $SCRIPT_DIR

if [ -f ../.env ]; then
  source ../.env
fi

# Load pyenv into the shell session.
# export PYENV_ROOT="$HOME/.pyenv"
# export PATH="$PYENV_ROOT/bin:$PATH"
# if command -v pyenv 1>/dev/null 2>&1; then
#   eval "$(pyenv init -)"
# fi

OUTPUT_LOG=output_etl.log
PID_FILE=etl_process.pid

# Function to find conda.sh
find_conda_sh() {
    # Attempt to locate conda.sh in common locations
    if [ -n "$CONDA_EXE" ]; then
        # CONDA_EXE is set; derive conda.sh path
        CONDA_PREFIX=$(dirname "$(dirname "$CONDA_EXE")")
        CONDA_SH="$CONDA_PREFIX/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
    else
        # Search for conda.sh in $HOME directory
        CONDA_SH=$(find "$HOME" -name conda.sh 2>/dev/null | head -n 1)
    fi
}

# Find conda.sh
find_conda_sh

# Check if conda.sh was found
if [ -f "$CONDA_SH" ]; then
    # Source conda.sh to enable 'conda' command
    source "$CONDA_SH"
else
    echo "Error: conda.sh not found" >> "$OUTPUT_LOG" 2>&1
    exit 1
fi

# Activate the Conda environment
conda activate "$ENV_NAME"

# COMMAND="$(pyenv which marten)"
COMMAND="$(which marten)"

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

# Deactivate the Conda environment
conda deactivate

# Write the PID to a file
echo $PID > $PID_FILE

# Schedule the monitoring script to run in 1 minute
at now + 1 minute <<EOF
bash $SCRIPT_DIR/monitor_etl.sh
EOF