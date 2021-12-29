#syntax=docker/dockerfile:1.1.5-experimental
ARG PY_VERSION
ARG IMAGE_TYPE

# Currenly all of our dev images are GPU capable but at a cost of being quite large.
# See https://github.com/tensorflow/build/pull/47
FROM tensorflow/build:latest-python$PY_VERSION as dev_container
ARG TF_PACKAGE
ARG TF_VERSION

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
