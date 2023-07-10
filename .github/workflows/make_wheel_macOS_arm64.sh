set -e -x

export TF_NEED_CUDA=0

python --version
python -m pip install --default-timeout=1000 delocate==0.10.3 wheel setuptools tensorflow==$TF_VERSION

python configure.py
# Setting DYLD_LIBRARY_PATH to help delocate finding tensorflow after the rpath invalidation
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$(python -c 'import configure; print(configure.get_tf_shared_lib_dir())')

# For dynamic linking, we want the ARM version of TensorFlow.
# Since we cannot run it on x86 so we need to force pip to install it regardless
python -m pip install \
  --platform=macosx_12_0_arm64 \
  --no-deps \
  --target=$(python -c 'import site; print(site.getsitepackages()[0])') \
  --upgrade \
  tensorflow-macos==$TF_VERSION

bazel build \
  --cpu=darwin_arm64 \
  --copt -mmacosx-version-min=12.0 \
  --linkopt -mmacosx-version-min=12.0 \
  --noshow_progress \
  --noshow_loading_progress \
  --verbose_failures \
  --test_output=errors \
  build_pip_pkg

bazel-bin/build_pip_pkg artifacts "--plat-name macosx_11_0_arm64 $NIGHTLY_FLAG"
delocate-wheel -w wheelhouse -v --ignore-missing-dependencies artifacts/*.whl

