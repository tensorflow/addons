set -e
bazel build //tensorflow_addons/...
cp ./bazel-bin/tensorflow_addons/custom_ops/activations/_*_ops.so ./tensorflow_addons/custom_ops/activations/
cp ./bazel-bin/tensorflow_addons/custom_ops/image/_*_ops.so ./tensorflow_addons/custom_ops/image/
cp ./bazel-bin/tensorflow_addons/custom_ops/layers/_*_ops.so ./tensorflow_addons/custom_ops/layers/
cp ./bazel-bin/tensorflow_addons/custom_ops/seq2seq/_*_ops.so ./tensorflow_addons/custom_ops/seq2seq/
cp ./bazel-bin/tensorflow_addons/custom_ops/text/_*_ops.so ./tensorflow_addons/custom_ops/text/
