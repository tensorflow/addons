set -e -x
wget --quiet -nc https://github.com/bazelbuild/bazel/releases/download/${1}/bazel-${1}-installer-darwin-x86_64.sh
chmod +x bazel-${1}-installer-darwin-x86_64.sh
./bazel-${1}-installer-darwin-x86_64.sh --user