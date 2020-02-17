FROM python:3.5-alpine as flake8-test

RUN pip install flake8==3.7.9
COPY ./ /addons
WORKDIR /addons
RUN flake8
RUN touch /ok.txt

# -------------------------------
FROM python:3.6 as black-test

RUN pip install black==19.10b0
COPY ./ /addons
RUN black --check /addons
RUN touch /ok.txt

# -------------------------------
FROM python:3.6 as public-api-typed

RUN pip install tensorflow-cpu==2.1.0
RUN pip install typeguard==2.7.1
RUN pip install typedapi==0.2.0

COPY ./ /addons
RUN TF_ADDONS_NO_BUILD=1 pip install --no-deps -e /addons
RUN python /addons/tools/ci_build/verify/check_typing_info.py
RUN touch /ok.txt

# -------------------------------
FROM python:3.5-alpine as case-insensitive-filesystem

COPY ./ /addons
WORKDIR /addons
RUN python /addons/tools/ci_build/verify/check_file_name.py
RUN touch /ok.txt

# -------------------------------
FROM python:3.5 as valid_build_files

RUN pip install tensorflow-cpu==2.1.0

RUN apt-get update && apt-get install sudo
RUN git clone https://github.com/abhinavsingh/setup-bazel.git
RUN bash ./setup-bazel/setup-bazel.sh 1.1.0

COPY ./ /addons
WORKDIR /addons
RUN python ./configure.py --no-deps
RUN bazel build --nobuild -- //tensorflow_addons/...
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

RUN wget -O /usr/local/bin/buildifier \
            https://github.com/bazelbuild/buildtools/releases/download/0.29.0/buildifier
RUN chmod +x /usr/local/bin/buildifier

COPY ./ /addons
RUN buildifier -mode=check -r /addons
RUN touch /ok.txt

# -------------------------------
# docs tests
FROM python:3.6 as docs_tests

RUN pip install tensorflow-cpu==2.1.0
RUN pip install typeguard==2.7.1

COPY tools/docs/doc_requirements.txt ./
RUN pip install -r doc_requirements.txt

RUN apt-get update && apt-get install -y rsync

COPY ./ /addons
WORKDIR /addons
RUN TF_ADDONS_NO_BUILD=1 pip install --no-deps -e .
RUN python tools/docs/build_docs.py
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
