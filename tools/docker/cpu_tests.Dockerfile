FROM python:3.5

RUN pip install tensorflow-cpu==2.1.0

RUN apt-get update && apt-get install sudo
RUN git clone https://github.com/abhinavsingh/setup-bazel.git
RUN bash ./setup-bazel/setup-bazel.sh 1.1.0

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY ./ /addons
WORKDIR addons
RUN bash tools/ci_testing/addons_cpu.sh --no-deps
