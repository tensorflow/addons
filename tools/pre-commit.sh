#!/usr/bin/env bash


python -m black --check ./
need_format=$?

set -e
if [ $need_format -ne 0 ]
then
    python -m black ./
    echo Some Python files were formatted
    echo You need to do git add and git commit again
    exit $need_format
fi

python -m flake8
