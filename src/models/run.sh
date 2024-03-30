#!/bin/sh

nohup `/home/jx/.pyenv/shims/python gridsearch.py --worker=10 --top_n=100 --epochs=500 930955 > /dev/null 2>&1` &
