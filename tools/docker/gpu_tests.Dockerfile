FROM tensorflow/tensorflow:2.1.0-custom-op-gpu-ubuntu16
ENV TF_NEED_CUDA="1"

ARG PY_VERSION
RUN ln -sf $(which python$PY_VERSION) /usr/bin/python

RUN python -m pip install --upgrade pip setuptools auditwheel==2.0.0

COPY tools/install_deps/tensorflow.txt ./
RUN python -m pip install -r tensorflow.txt

COPY requirements.txt ./
RUN python -m pip install -r requirements.txt


COPY ./ /addons
WORKDIR addons
CMD ["bash", "tools/testing/build_and_run_tests.sh"]
