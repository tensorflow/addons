set -e -x

if [ "$TF_NEED_CUDA" == "1" ]; then
  CUDA_FLAG="--crosstool_top=@ubuntu20.04-gcc9_manylinux2014-cuda11.2-cudnn8.1-tensorrt7.2_config_cuda//crosstool:toolchain"
fi

bazel build $CUDA_FLAG //tensorflow_addons/...
cp ./bazel-bin/tensorflow_addons/custom_ops/image/_*_ops.so ./tensorflow_addons/custom_ops/image/
cp ./bazel-bin/tensorflow_addons/custom_ops/layers/_*_ops.so ./tensorflow_addons/custom_ops/layers/
cp ./bazel-bin/tensorflow_addons/custom_ops/seq2seq/_*_ops.so ./tensorflow_addons/custom_ops/seq2seq/
cp ./bazel-bin/tensorflow_addons/custom_ops/text/_*_ops.so ./tensorflow_addons/custom_ops/text/
cp ./bazel-bin/tensorflow_addons/custom_ops/text/_parse_time_op.so ./tensorflow_addons/custom_ops/text/
