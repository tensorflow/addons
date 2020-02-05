# Flake8
FROM python:3.5

RUN pip install flake8==3.7.9
COPY ./ /addons
WORKDIR /addons
RUN flake8
RUN touch /ok.txt

# -------------------------------
# Black Python code format
FROM python:3.5

RUN pip install black==19.10b0
COPY ./ /addons
RUN black --check /addons
RUN touch /ok.txt

# -------------------------------
# Check that the public API is typed
FROM python:3.5

RUN pip install tensorflow-cpu==2.1.0 typeguard==2.7.1
COPY ./ /addons
RUN TF_ADDONS_NO_BUILD=1 pip install --no-deps -e /addons
RUN python /addons/tools/ci_build/verify/check_typing_info.py
RUN touch /ok.txt

# -------------------------------
# Verify python filenames work on case insensitive FS
FROM python:3.5

COPY ./ /addons
WORKDIR /addons
RUN python /addons/tools/ci_build/verify/check_file_name.py
RUN touch /ok.txt

# -------------------------------
# Valid build files
FROM python:3.5

RUN apt-get update && apt-get install sudo
RUN git clone https://github.com/abhinavsingh/setup-bazel.git
RUN bash ./setup-bazel/setup-bazel.sh 1.1.0

RUN pip install tensorflow-cpu==2.1.0

COPY ./ /addons
WORKDIR /addons
RUN python ./configure.py --no-deps
RUN bazel build --nobuild -- //tensorflow_addons/...
RUN touch /ok.txt

# -------------------------------
# Clang C++ code format
FROM python:3.5

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
FROM alpine:3.11

RUN wget -O /usr/local/bin/buildifier \
            https://github.com/bazelbuild/buildtools/releases/download/0.29.0/buildifier
RUN chmod +x /usr/local/bin/buildifier

COPY ./ /addons
RUN buildifier -mode=check -r /addons
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
