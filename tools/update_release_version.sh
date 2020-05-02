#!/usr/bin/env bash
# usage: bash tools/update_release_version.sh <release_number>

sed -ri "s/(TF_VERSION=|tensorflow(-cpu)*(~|=)=|tf-version: \[')[0-9]+[a-zA-Z0-9_.-]+/\1$1/g" \
	.github/workflows/release.yml \
	CONTRIBUTING.md \
	tools/docker/cpu_tests.Dockerfile \
	tools/install_deps/tensorflow-cpu.txt \
	tools/install_deps/tensorflow.txt \
	tools/run_gpu_tests.sh
