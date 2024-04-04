#!/bin/bash

script=gridsearch.py
PYENV_PYTHON=$(pyenv which python)
>grid_search.log
>output.log
# nohup "$PYENV_PYTHON" "$script" "--worker=14" "--top_n=100" "--epochs=500" "930955" > /dev/null 2>&1 &
nohup "$PYENV_PYTHON" "$script" "$@" > >(tee -a output.log) 2>&1 &
