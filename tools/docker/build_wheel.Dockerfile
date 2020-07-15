#syntax=docker/dockerfile:1.1.5-experimental
ARG TF_VERSION
ARG PY_VERSION
FROM tfaddons/tensorflow:2.1.0-custom-op-gpu-ubuntu16-minimal as base_install
ENV TF_NEED_CUDA="1"

# is needed because when we sqashed the image, we lost all environment variables.
ENV NVIDIA_REQUIRE_CUDA=cuda>=10.1 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs
ENV NVIDIA_VISIBLE_DEVICES=all
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN apt-get update && apt-get install patchelf

# Fix presented in
# https://stackoverflow.com/questions/44967202/pip-is-showing-error-lsb-release-a-returned-non-zero-exit-status-1/44967506
RUN echo "#! /usr/bin/python2.7" >> /usr/bin/lsb_release2
RUN cat /usr/bin/lsb_release >> /usr/bin/lsb_release2
RUN mv /usr/bin/lsb_release2 /usr/bin/lsb_release

ARG PY_VERSION
RUN ln -sf $(which python$PY_VERSION) /usr/bin/python

RUN python -m pip install setuptools

RUN python -m pip install --upgrade pip==19.0 auditwheel==2.0.0

ARG TF_VERSION
RUN python -m pip install --default-timeout=1000 tensorflow==$TF_VERSION

COPY tools/install_deps/ /install_deps
RUN python -m pip install -r /install_deps/pytest.txt

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

FROM python:$PY_VERSION as test_wheel_in_fresh_environment

ARG TF_VERSION
RUN python -m pip install --default-timeout=1000 tensorflow==$TF_VERSION

COPY --from=make_wheel /addons/wheelhouse/ /addons/wheelhouse/
RUN pip install /addons/wheelhouse/*.whl

RUN python -c "import tensorflow_addons as tfa; print(tfa.register_all())"

# -------------------------------------------------------------------
FROM scratch as output

COPY --from=test_wheel_in_fresh_environment /addons/wheelhouse/ .
