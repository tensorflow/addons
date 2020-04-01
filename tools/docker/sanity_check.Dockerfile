#syntax=docker/dockerfile:1.1.5-experimental
FROM python:3.5-alpine as flake8-test

COPY tools/install_deps/flake8.txt ./
RUN pip install -r flake8.txt
COPY ./ /addons
WORKDIR /addons
RUN flake8
RUN touch /ok.txt

# -------------------------------
FROM python:3.6 as black-test

COPY tools/install_deps/black.txt ./
RUN pip install -r black.txt
COPY ./ /addons
RUN black --check /addons
RUN touch /ok.txt

# -------------------------------
FROM python:3.6 as public-api-typed

COPY tools/install_deps/tensorflow-cpu.txt ./
RUN pip install -r tensorflow-cpu.txt
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY tools/install_deps/typedapi.txt ./
RUN pip install -r typedapi.txt


COPY ./ /addons
RUN pip install --no-deps -e /addons
RUN python /addons/tools/testing/check_typing_info.py
RUN touch /ok.txt

# -------------------------------
FROM python:3.5-alpine as case-insensitive-filesystem

COPY ./ /addons
WORKDIR /addons
RUN python /addons/tools/testing/check_file_name.py
RUN touch /ok.txt

# -------------------------------
FROM python:3.5 as valid_build_files

COPY tools/install_deps/tensorflow-cpu.txt ./
RUN pip install -r tensorflow-cpu.txt

RUN apt-get update && apt-get install sudo
COPY tools/install_deps/bazel_linux.sh ./
RUN bash bazel_linux.sh

COPY ./ /addons
WORKDIR /addons
RUN python ./configure.py
RUN --mount=type=cache,id=cache_bazel,target=/root/.cache/bazel \
    bazel build --nobuild -- //tensorflow_addons/...
RUN touch /ok.txt

# -------------------------------
FROM python:3.6-alpine as clang-format

RUN apk add --no-cache git
RUN git clone https://github.com/gabrieldemarmiesse/clang-format-lint-action.git
WORKDIR ./clang-format-lint-action
RUN git checkout 1044fee

COPY ./ /addons
RUN python run-clang-format.py \
               -r \
               --cli-args=--style=google \
               --clang-format-executable ./clang-format/clang-format9 \
               /addons
RUN touch /ok.txt

# -------------------------------
# Bazel code format
FROM alpine:3.11 as check-bazel-format

COPY ./tools/install_deps/buildifier.sh ./
RUN sh buildifier.sh

COPY ./ /addons
RUN buildifier -mode=check -r /addons
RUN touch /ok.txt

# -------------------------------
# docs tests
FROM python:3.6 as docs_tests

COPY tools/install_deps/tensorflow-cpu.txt ./
RUN pip install -r tensorflow-cpu.txt
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY tools/install_deps/doc_requirements.txt ./
RUN pip install -r doc_requirements.txt

RUN apt-get update && apt-get install -y rsync

COPY ./ /addons
WORKDIR /addons
RUN pip install --no-deps -e .
RUN python docs/build_docs.py
RUN touch /ok.txt

# -------------------------------
# test the editable mode
FROM python:3.6 as test_editable_mode

COPY tools/install_deps/tensorflow-cpu.txt ./
RUN pip install -r tensorflow-cpu.txt
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY tools/install_deps/pytest.txt ./
RUN pip install -r pytest.txt

RUN apt-get update && apt-get install -y sudo rsync
COPY tools/install_deps/bazel_linux.sh ./
RUN bash bazel_linux.sh

COPY ./ /addons
WORKDIR /addons
RUN python configure.py
RUN bash tools/install_so_files.sh
RUN pip install --no-deps -e .
RUN pytest -v -n auto ./tensorflow_addons/activations
RUN touch /ok.txt

# -------------------------------
# ensure that all checks were successful
# this is necessary if using docker buildkit
# with "export DOCKER_BUILDKIT=1"
# otherwise dead branch elimination doesn't
# run all tests
FROM scratch

COPY --from=0 /ok.txt /ok0.txt
COPY --from=1 /ok.txt /ok1.txt
COPY --from=2 /ok.txt /ok2.txt
COPY --from=3 /ok.txt /ok3.txt
COPY --from=4 /ok.txt /ok4.txt
COPY --from=5 /ok.txt /ok5.txt
COPY --from=6 /ok.txt /ok6.txt
COPY --from=7 /ok.txt /ok7.txt
COPY --from=8 /ok.txt /ok8.txt
