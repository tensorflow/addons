set -e
export WORKDIR=$(pwd)
mkdir /tmp/tf_addons_84d6d48
cd /tmp/tf_addons_84d6d48
git clone https://github.com/abhinavsingh/setup-bazel.git
bash ./setup-bazel/setup-bazel.sh 1.1.0
rm -rf /tmp/tf_addons_84d6d48
cd "$WORKDIR"
