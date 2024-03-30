#!/bin/bash

script=gridsearch.py
PYENV_PYTHON=$(pyenv which python)

nohup "$PYENV_PYTHON" "$script" "--worker=10" "--top_n=100" "--epochs=500" "930955" > /dev/null 2>&1 &
