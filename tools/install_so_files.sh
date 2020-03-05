set -e
bazel build //tensorflow_addons/...
cp ./bazel-bin/tensorflow_addons/custom_ops/activations/*.so ./tensorflow_addons/custom_ops/activations/
cp ./bazel-bin/tensorflow_addons/custom_ops/image/*.so ./tensorflow_addons/custom_ops/image/
cp ./bazel-bin/tensorflow_addons/custom_ops/layers/*.so ./tensorflow_addons/custom_ops/layers/
cp ./bazel-bin/tensorflow_addons/custom_ops/seq2seq/*.so ./tensorflow_addons/custom_ops/seq2seq/
cp ./bazel-bin/tensorflow_addons/custom_ops/text/*.so ./tensorflow_addons/custom_ops/text/
