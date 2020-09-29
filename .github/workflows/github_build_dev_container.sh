#!/usr/bin/env bash

set -x -e

df -h
docker info
# to get more disk space
rm -rf /usr/share/dotnet &

tools/build_dev_container.sh
