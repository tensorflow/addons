#syntax=docker/dockerfile:1.1.5-experimental
FROM addons_base:latest as build_dev_container

COPY tools/install_deps /install_deps
RUN pip install -r /install_deps/black.txt -r /install_deps/flake8.txt
RUN bash /install_deps/buildifier.sh
RUN bash /install_deps/clang-format.sh

ENV ADDONS_DEV_CONTAINER="1"

# Clean up
RUN apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*