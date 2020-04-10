#syntax=docker/dockerfile:1.1.5-experimental
ARG TF_VERSION
FROM seanpmorgan/tensorflow:2.1.0-custom-op-gpu-ubuntu16-minimal as base_install
ENV TF_NEED_CUDA="1"

RUN apt-get update && apt-get install patchelf

ARG PY_VERSION

RUN ln -sf $(which python$PY_VERSION) /usr/bin/python3

RUN python3 -m pip install --upgrade pip setuptools auditwheel==2.0.0

COPY tools/install_deps/ /install_deps
ARG TF_VERSION
RUN python3 -m pip install \
        tensorflow==$TF_VERSION \
        -r /install_deps/pytest.txt

COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

# Fix presented in
# https://stackoverflow.com/questions/44967202/pip-is-showing-error-lsb-release-a-returned-non-zero-exit-status-1/44967506
RUN echo "#! /usr/bin/python2.7" >> /usr/bin/lsb_release2
RUN cat /usr/bin/lsb_release >> /usr/bin/lsb_release2
RUN mv /usr/bin/lsb_release2 /usr/bin/lsb_release

COPY ./ /addons
WORKDIR /addons

# -------------------------------------------------------------------
FROM base_install as tfa_gpu_tests
CMD ["bash", "tools/testing/build_and_run_tests.sh"]

# -------------------------------------------------------------------
FROM base_install as make_wheel
ARG NIGHTLY_FLAG
ARG NIGHTLY_TIME
RUN --mount=type=cache,id=cache_bazel,target=/root/.cache/bazel \
    bash tools/testing/build_and_run_tests.sh && \
    bash tools/releases/release_linux.sh $NIGHTLY_FLAG

RUN bash tools/releases/tf_auditwheel_patch.sh
RUN auditwheel repair --plat manylinux2010_x86_64 artifacts/*.whl
RUN ls -al wheelhouse/

# -------------------------------------------------------------------
FROM scratch as output

COPY --from=make_wheel /addons/wheelhouse/ .
