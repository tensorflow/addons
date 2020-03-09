set -e -x

docker run -e TF_NEED_CUDA=1 -v ${PWD}:/addons -w /addons \
  tensorflow/tensorflow:2.1.0-custom-op-gpu-ubuntu16 \
  tools/releases/release_linux.sh $PY_VERSION $NIGHTLY_FLAG

sudo apt-get install patchelf
python3 -m pip install -U auditwheel==2.0.0
tools/releases/tf_auditwheel_patch.sh

auditwheel repair --plat manylinux2010_x86_64 artifacts/*.whl

ls -al wheelhouse/
