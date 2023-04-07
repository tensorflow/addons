#syntax=docker/dockerfile:1.1.5-experimental
FROM python:3.9 as build_wheel

ARG TF_VERSION=2.12.0
RUN pip install --default-timeout=1000 tensorflow-cpu==$TF_VERSION

RUN apt-get update && apt-get install -y sudo rsync
COPY tools/install_deps/install_bazelisk.sh .bazeliskrc ./
RUN bash install_bazelisk.sh

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY tools/install_deps/pytest.txt ./
RUN pip install -r pytest.txt pytest-cov

COPY ./ /addons
WORKDIR addons
RUN python configure.py
RUN pip install -e ./
RUN --mount=type=cache,id=cache_bazel,target=/root/.cache/bazel \
    bash tools/install_so_files.sh
RUN pytest -v -n auto --durations=25 --doctest-modules ./tensorflow_addons \
    --cov=tensorflow_addons ./tensorflow_addons/

RUN bazel build --enable_runfiles build_pip_pkg
RUN bazel-bin/build_pip_pkg artifacts


FROM python:3.9

COPY tools/install_deps/tensorflow-cpu.txt ./
RUN pip install --default-timeout=1000 -r tensorflow-cpu.txt

COPY --from=0 /addons/artifacts /artifacts

RUN pip install /artifacts/tensorflow_addons-*.whl

# check that we didnd't forget to add a py file to
# The corresponding BUILD file.
# Also test that the wheel works in a fresh environment
RUN python -c "import tensorflow_addons as tfa; print(tfa.register_all())"
