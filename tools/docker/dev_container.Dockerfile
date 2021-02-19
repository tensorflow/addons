#syntax=docker/dockerfile:1.1.5-experimental
FROM tensorflow/tensorflow:2.1.0-custom-op-ubuntu16 as dev_container_cpu
ARG TF_PACKAGE
ARG TF_VERSION

# Temporary until custom-op container is updated
RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN ln -sf /usr/local/bin/pip3 /usr/local/bin/pip
RUN pip install --default-timeout=1000 $TF_PACKAGE==$TF_VERSION

COPY tools/install_deps /install_deps
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /install_deps/black.txt \
    -r /install_deps/flake8.txt \
    -r /install_deps/pytest.txt \
    -r /install_deps/typedapi.txt \
    -r /tmp/requirements.txt

RUN bash /install_deps/buildifier.sh
RUN bash /install_deps/clang-format.sh

ENV ADDONS_DEV_CONTAINER="1"

# Clean up
RUN apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*
