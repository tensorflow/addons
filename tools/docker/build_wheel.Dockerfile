#syntax=docker/dockerfile:1.1.5-experimental
ARG PY_VERSION
FROM tensorflow/build:latest-python$PY_VERSION as base_install

ENV TF_NEED_CUDA="1"
ARG PY_VERSION
ARG TF_VERSION
RUN python -m pip install --default-timeout=1000 tensorflow==$TF_VERSION

COPY tools/install_deps/ /install_deps
RUN python -m pip install -r /install_deps/pytest.txt

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

COPY ./ /addons
WORKDIR /addons

# -------------------------------------------------------------------K
FROM base_install as tfa_gpu_tests
CMD ["bash", "tools/testing/build_and_run_tests.sh"]

# -------------------------------------------------------------------
FROM base_install as make_wheel
ARG NIGHTLY_FLAG
ARG NIGHTLY_TIME

RUN python configure.py

RUN bash tools/testing/build_and_run_tests.sh && \
    bazel build \
        --noshow_progress \
        --noshow_loading_progress \
        --verbose_failures \
        --test_output=errors \
        --crosstool_top=//build_deps/toolchains/gcc7_manylinux2010-nvcc-cuda11:toolchain \
        build_pip_pkg && \
    # Package Whl
    bazel-bin/build_pip_pkg artifacts $NIGHTLY_FLAG

RUN bash tools/releases/tf_auditwheel_patch.sh
RUN python -m auditwheel repair --plat manylinux2010_x86_64 artifacts/*.whl
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
