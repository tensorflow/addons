set -e -x

export TF_NEED_CUDA=0

# Install Deps
python --version
python -m pip install --default-timeout=1000 delocate==0.9.1 wheel setuptools tensorflow==$TF_VERSION

# Test
bash ./tools/testing/build_and_run_tests.sh

# Clean
bazel clean

# Build
python configure.py

bazel build \
  --copt -mmacosx-version-min=10.13 \
  --linkopt -mmacosx-version-min=10.13 \
  --noshow_progress \
  --noshow_loading_progress \
  --verbose_failures \
  --test_output=errors \
  build_pip_pkg

bazel-bin/build_pip_pkg artifacts $NIGHTLY_FLAG
delocate-wheel -w wheelhouse artifacts/*.whl

