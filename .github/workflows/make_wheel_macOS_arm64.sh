set -e -x

export TF_NEED_CUDA=0

python --version
python -m pip install --default-timeout=1000 delocate wheel setuptools tensorflow==$TF_VERSION

python configure.py

bazel build \
  --cpu=darwin_arm64 \
  --copt -mmacosx-version-min=11.0 \
  --linkopt -mmacosx-version-min=11.0 \
  --noshow_progress \
  --noshow_loading_progress \
  --verbose_failures \
  --test_output=errors \
  build_pip_pkg

bazel-bin/build_pip_pkg artifacts "--plat-name macosx_11_0_arm64 $NIGHTLY_FLAG"
delocate-wheel -w wheelhouse artifacts/*.whl

