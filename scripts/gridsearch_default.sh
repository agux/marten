#!/bin/bash

CLI="$(pyenv which marten)"
OUTPUT_LOG=output_gs.log
# PYENV_PYTHON=$(pyenv which python)

>grid_search.log
>$OUTPUT_LOG

# nohup "$PYENV_PYTHON" "$script" "--worker=14" "--top_n=100" "--epochs=500" "930955" > /dev/null 2>&1 &
nohup "$CLI" "gs" "$@" > >(tee -a $OUTPUT_LOG) 2>&1 &
