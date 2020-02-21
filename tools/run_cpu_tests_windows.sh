set -e -x
python -m pip install wheel setuptools
curl -sSOL https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-windows-x86_64.exe
export BAZEL_PATH=/d/a/addons/addons/bazel-${BAZEL_VERSION}-windows-x86_64.exe
bash ./tools/ci_testing/addons_cpu.sh
