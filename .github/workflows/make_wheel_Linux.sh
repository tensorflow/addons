set -e -x

 docker run -e TF_NEED_CUDA=1 -v ${PWD}:/addons -w /addons \
  tensorflow/tensorflow:2.1.0-custom-op-gpu-ubuntu16 \
  tools/ci_build/builds/release_linux.sh $PY_VERSION $NIGHTLY_FLAG

sudo apt-get install patchelf
python3 -m pip install -U auditwheel==2.0.0
tools/ci_build/builds/tf_auditwheel_patch.sh

for f in artifacts/*.whl; do
  auditwheel repair --plat manylinux2010_x86_64 $f
done
ls -al wheelhouse/
