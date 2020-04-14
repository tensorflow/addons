#syntax=docker/dockerfile:1.1.5-experimental
ARG TF_VERSION
FROM seanpmorgan/tensorflow:2.1.0-custom-op-gpu-ubuntu16-minimal as base_install
ENV TF_NEED_CUDA="1"

RUN apt-get update && apt-get install patchelf

# Fix presented in
# https://stackoverflow.com/questions/44967202/pip-is-showing-error-lsb-release-a-returned-non-zero-exit-status-1/44967506
RUN echo "#! /usr/bin/python2.7" >> /usr/bin/lsb_release2
RUN cat /usr/bin/lsb_release >> /usr/bin/lsb_release2
RUN mv /usr/bin/lsb_release2 /usr/bin/lsb_release

ARG PY_VERSION
RUN ln -sf $(which python$PY_VERSION) /usr/bin/python

RUN python -m pip install --upgrade pip setuptools auditwheel==2.0.0

COPY tools/install_deps/ /install_deps
ARG TF_VERSION
RUN python -m pip install --no-cache-dir \
        tensorflow==$TF_VERSION \
        -r /install_deps/pytest.txt

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

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
    bazel clean --expunge && \
    bazel build \
        -c opt \
        --noshow_progress \
        --noshow_loading_progress \
        --verbose_failures \
        --test_output=errors \
        --crosstool_top=//build_deps/toolchains/gcc7_manylinux2010-nvcc-cuda10.1:toolchain \
        build_pip_pkg && \
    # Package Whl
    bazel-bin/build_pip_pkg artifacts $NIGHTLY_FLAG

RUN bash tools/releases/tf_auditwheel_patch.sh
RUN auditwheel repair --plat manylinux2010_x86_64 artifacts/*.whl
RUN ls -al wheelhouse/

# -------------------------------------------------------------------
FROM python:3.5 as test_wheel_in_fresh_environement

ARG TF_VERSION
RUN python -m pip install --no-cache-dir tensorflow==$TF_VERSION

COPY --from=make_wheel /addons/wheelhouse/ /addons/wheelhouse/
RUN pip install /addons/wheelhouse/*.whl

RUN python -c "import tensorflow_addons as tfa; print(tfa.activations.lisht(0.2))"

# -------------------------------------------------------------------
FROM scratch as output

COPY --from=test_wheel_in_fresh_environement /addons/wheelhouse/ .
