FROM python:3.5

RUN pip install tensorflow-cpu==2.1.0

RUN apt-get update && apt-get install sudo
COPY tools/ci_build/install/bazel.sh ./
RUN bash bazel.sh

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY ./ /addons
WORKDIR addons
RUN bash tools/ci_testing/addons_cpu.sh --no-deps
