set -e -x

export TF_NEED_CUDA=0
export BAZEL_VC="C:/Program Files (x86)/Microsoft Visual Studio/2019/Enterprise/VC/"

python -m pip install --default-timeout=1000 wheel setuptools tensorflow==$TF_VERSION
bash ./tools/testing/build_and_run_tests.sh

bazel.exe build \
  -c opt \
  --enable_runfiles \
  --noshow_progress \
  --noshow_loading_progress \
  --verbose_failures \
  --test_output=errors \
  build_pip_pkg
bazel-bin/build_pip_pkg wheelhouse $NIGHTLY_FLAG
