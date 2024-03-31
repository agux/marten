#!/bin/bash

script=etl.py
PYENV_PYTHON=$(pyenv which python)

nohup "$PYENV_PYTHON" "$script" "$@" > /dev/null 2>&1 &
