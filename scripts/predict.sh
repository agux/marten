#!/bin/bash

CLI="$(pyenv which marten)"
OUTPUT_LOG=predict.log
# PYENV_PYTHON=$(pyenv which python)

>$OUTPUT_LOG
>marten.models.predict.log
>marten.utils.logger.log

# Clear old lr finder .ckpt files
rm -f .lr_find_*.ckpt
rm -rf lightning_logs

export MALLOC_TRIM_THRESHOLD_=0
# export DASK_DISTRIBUTED__WORKER__RESOURCES__POWER=2

nohup "$CLI" "predict" "$@" > >(tee -a $OUTPUT_LOG) 2>&1 &
