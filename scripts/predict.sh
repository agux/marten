#!/bin/bash

CLI="$(pyenv which marten)"
OUTPUT_LOG=predict.log
# PYENV_PYTHON=$(pyenv which python)

>$OUTPUT_LOG
>marten.models.predict.log

# nohup "$PYENV_PYTHON" "$script" "--worker=14" "--top_n=100" "--epochs=500" "930955" > /dev/null 2>&1 &
nohup "$CLI" "predict" "$@" > >(tee -a $OUTPUT_LOG) 2>&1 &
