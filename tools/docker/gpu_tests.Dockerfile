FROM tensorflow/tensorflow:2.1.0-custom-op-gpu-ubuntu16

ARG PY_VERSION
RUN python$PY_VERSION -m pip install --upgrade pip setuptools auditwheel==2.0.0

COPY tools/install_deps/ /install_deps
RUN python$PY_VERSION -m pip install \
        -r /install_deps/tensorflow.txt \
        -r /install_deps/pytest.txt

COPY requirements.txt .
RUN python$PY_VERSION -m pip install -r requirements.txt

COPY ./ /addons
WORKDIR addons
CMD ["bash", "tools/testing/addons_gpu.sh"]
