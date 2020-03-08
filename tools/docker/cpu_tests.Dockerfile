FROM python:3.5

RUN apt-get update && apt-get install -y sudo rsync

COPY tools/install_deps /install_deps
RUN pip install -r /install_deps/tensorflow-cpu.txt
RUN bash /install_deps/bazel_linux.sh
RUN bash /install_deps/finish_bazel_install.sh

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY ./ /addons
WORKDIR addons
RUN bash tools/ci_testing/addons_cpu.sh --no-deps

RUN bazel build --enable_runfiles build_pip_pkg
RUN bazel-bin/build_pip_pkg artifacts


FROM python:3.5

COPY tools/install_deps /install_deps
RUN pip install -r /install_deps/tensorflow-cpu.txt

COPY --from=0 /addons/artifacts /artifacts

RUN pip install /artifacts/tensorflow_addons-*.whl

# check that we didnd't forget to add a py file to
# The corresponding BUILD file.
# Also test that the wheel works in a fresh environment
RUN python -c "import tensorflow_addons"
