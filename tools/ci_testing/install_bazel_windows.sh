set -e -x
curl -sSOL https://github.com/bazelbuild/bazel/releases/download/${1}/bazel-${1}-windows-x86_64.exe
echo export BAZEL_VC=/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2019/Enterprise/VC/  >> ~/.bash_profile
echo export BAZEL_PATH=/d/a/addons/addons/bazel-${1}-windows-x86_64.exe  >> ~/.bash_profile
echo export PATH=/c/hostedtoolcache/windows/Python/${2}/x64/:$PATH  >> ~/.bash_profile
