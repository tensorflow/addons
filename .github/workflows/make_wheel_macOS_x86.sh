set -e -x

export TF_NEED_CUDA=0

# Install Deps
python --version
python -m pip install --default-timeout=1000 delocate==0.10.3 wheel setuptools tensorflow==$TF_VERSION

# Test
bash ./tools/testing/build_and_run_tests.sh $SKIP_CUSTOM_OP_TESTS

# Clean
bazel clean

# Build
python configure.py

bazel build \
  --copt=-mmacosx-version-min=10.14 \
  --linkopt=-mmacosx-version-min=10.14 \
  --noshow_progress \
  --noshow_loading_progress \
  --verbose_failures \
  --test_output=errors \
  build_pip_pkg

bazel-bin/build_pip_pkg artifacts $NIGHTLY_FLAG

# Setting DYLD_LIBRARY_PATH to help delocate finding tensorflow after the rpath invalidation
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$(python -c 'import configure; print(configure.get_tf_shared_lib_dir())')
delocate-wheel -w wheelhouse -v --ignore-missing-dependencies artifacts/*.whl

