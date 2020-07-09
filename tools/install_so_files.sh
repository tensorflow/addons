set -e -x

if [ "$TF_NEED_CUDA" == "1" ]; then
  CUDA_FLAG="--crosstool_top=//build_deps/toolchains/gcc7_manylinux2010-nvcc-cuda10.1:toolchain"
fi

bazel build $CUDA_FLAG //tensorflow_addons/...
cp ./bazel-bin/tensorflow_addons/custom_ops/activations/_*_ops.so ./tensorflow_addons/custom_ops/activations/
cp ./bazel-bin/tensorflow_addons/custom_ops/image/_*_ops.so ./tensorflow_addons/custom_ops/image/
cp ./bazel-bin/tensorflow_addons/custom_ops/layers/_*_ops.so ./tensorflow_addons/custom_ops/layers/
cp ./bazel-bin/tensorflow_addons/custom_ops/seq2seq/_*_ops.so ./tensorflow_addons/custom_ops/seq2seq/
cp ./bazel-bin/tensorflow_addons/custom_ops/text/_*_ops.so ./tensorflow_addons/custom_ops/text/
cp ./bazel-bin/tensorflow_addons/custom_ops/text/_parse_time_op.so ./tensorflow_addons/custom_ops/text/
