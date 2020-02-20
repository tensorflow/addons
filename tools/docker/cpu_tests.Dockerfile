FROM python:3.5

RUN pip install tensorflow-cpu==2.1.0

RUN apt-get update && apt-get install -y sudo rsync
COPY tools/ci_build/install/bazel.sh ./
RUN bash bazel.sh

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY ./ /addons
WORKDIR addons
RUN bash tools/ci_testing/addons_cpu.sh --no-deps

RUN bazel build --enable_runfiles build_pip_pkg
RUN bazel-bin/build_pip_pkg artifacts


FROM python:3.5

RUN pip install tensorflow-cpu==2.1.0

COPY --from=0 /addons/artifacts /artifacts

RUN pip install /artifacts/tensorflow_addons-*.whl

# check that we didnd't forget to add a py file to
# The corresponding BUILD file.
# Also test that the wheel works in a fresh environment
RUN python -c "import tensorflow_addons"
