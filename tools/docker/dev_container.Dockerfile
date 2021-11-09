#syntax=docker/dockerfile:1.1.5-experimental
FROM gcr.io/tensorflow-testing/nosla-cuda11.2-cudnn8.1-ubuntu18.04-manylinux2010-multipython as dev_container_cpu
ARG TF_PACKAGE
ARG TF_VERSION

# Temporary until custom-op container is updated
RUN ln -sf /usr/local/bin/python3.8 /usr/bin/python
RUN pip3.8 install --default-timeout=1000 $TF_PACKAGE==$TF_VERSION

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
