#!/bin/bash

bazel test -c opt -k \
      --test_timeout 300,450,1200,3600 \
      --test_output=all \
      --run_under=$(readlink -f tools/ci_testing/parallel_gpu_execute.sh) \
      //tensorflow_addons/layers:crf_test

bazel test -c opt -k \
      --test_timeout 300,450,1200,3600 \
      --test_output=all \
      --run_under=$(readlink -f tools/ci_testing/parallel_gpu_execute.sh) \
      //tensorflow_addons/losses:crf_test
