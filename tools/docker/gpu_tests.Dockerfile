FROM tensorflow/tensorflow:2.1.0-custom-op-gpu-ubuntu16
ENV TF_NEED_CUDA="1"
ENV TF_CUDA_VERSION="10.1"
ENV CUDA_TOOLKIT_PATH="/usr/local/cuda"
ENV TF_CUDNN_VERSION="7"
ENV CUDNN_INSTALL_PATH="/usr/lib/x86_64-linux-gnu"

RUN python3 -m pip install --upgrade pip setuptools auditwheel==2.0.0

COPY tools/install_deps/tensorflow.txt ./
RUN python3 -m pip install -r tensorflow.txt

COPY requirements.txt ./
RUN python3 -m pip install -r requirements.txt


COPY ./ /addons
WORKDIR addons
CMD ["bash", "tools/testing/build_and_run_tests.sh"]
