#!/bin/bash

CLI="$(pyenv which marten)"
OUTPUT_LOG=predict.log
# PYENV_PYTHON=$(pyenv which python)

>$OUTPUT_LOG
>marten.models.predict.log

export MALLOC_TRIM_THRESHOLD_=0

nohup "$CLI" "predict" "$@" > >(tee -a $OUTPUT_LOG) 2>&1 &
