set -e -x

export TF_NEED_CUDA=0

python3 --version
python3 -m pip install delocate wheel setuptools

bash tools/install_deps/bazel_macos.sh $BAZEL_VERSION
bash tools/ci_testing/addons_cpu.sh

bazel build \
  -c opt \
  --copt -mmacosx-version-min=10.13 \
  --linkopt -mmacosx-version-min=10.13 \
  --noshow_progress \
  --noshow_loading_progress \
  --verbose_failures \
  --test_output=errors \
  build_pip_pkg

bazel-bin/build_pip_pkg artifacts $NIGHTLY_FLAG
delocate-wheel -w wheelhouse artifacts/*.whl

