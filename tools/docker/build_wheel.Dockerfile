#syntax=docker/dockerfile:1.1.5-experimental
ARG TF_VERSION
FROM seanpmorgan/tensorflow:2.1.0-custom-op-gpu-ubuntu16-minimal as make_wheel
ENV TF_NEED_CUDA="1"

RUN apt-get update && apt-get install patchelf

ARG PY_VERSION
RUN python$PY_VERSION -m pip install --upgrade pip setuptools auditwheel==2.0.0

COPY tools/install_deps/ /install_deps
ARG TF_VERSION
RUN python$PY_VERSION -m pip install \
        tensorflow==$TF_VERSION \
        -r /install_deps/pytest.txt

COPY requirements.txt .
RUN python$PY_VERSION -m pip install -r requirements.txt

COPY ./ /addons
WORKDIR /addons
ARG NIGHTLY_FLAG
ARG NIGHTLY_TIME
RUN --mount=type=cache,id=cache_bazel,target=/root/.cache/bazel \
    bash tools/releases/release_linux.sh $PY_VERSION $NIGHTLY_FLAG

RUN bash tools/releases/tf_auditwheel_patch.sh
RUN auditwheel repair --plat manylinux2010_x86_64 artifacts/*.whl
RUN ls -al wheelhouse/

FROM scratch as output

COPY --from=make_wheel /addons/wheelhouse/ .
