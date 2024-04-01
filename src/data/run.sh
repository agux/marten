#!/bin/bash

script=etl.py
PYENV_PYTHON=$(pyenv which python)
>etl.log
>output.log
nohup "$PYENV_PYTHON" "$script" "$@" > >(tee -a output.log) 2>&1 &
