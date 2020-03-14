FROM tensorflow/tensorflow:2.1.0-custom-op-gpu-ubuntu16 as base

COPY tools/install_deps/tensorflow.txt ./

RUN python3 -m pip install -r tensorflow.txt

COPY requirements.txt ./
RUN python3 -m pip install -r requirements.txt

COPY tools/install_deps/finish_bazel_install.sh ./
RUN bash finish_bazel_install.sh

FROM base as gpu_tests
COPY ./ /addons
WORKDIR addons
CMD ["bash", "tools/testing/addons_gpu.sh"]

FROM base as interactive_dev
# This Dockerfile adds a non-root user with sudo access. Use the "remoteUser"
# property in devcontainer.json to use it. On Linux, the container user's GID/UIDs
# will be updated to match your local UID/GID (when using the dockerFile property).
# See https://aka.ms/vscode-remote/containers/non-root-user for details.
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Set to false to skip installing zsh and Oh My ZSH!
ARG INSTALL_ZSH="false"

# Location and expected SHA for common setup script - SHA generated on release
ARG COMMON_SCRIPT_SOURCE="https://raw.githubusercontent.com/microsoft/vscode-dev-containers/master/script-library/common-debian.sh"
ARG COMMON_SCRIPT_SHA="dev-mode"

RUN apt-get update \
  && apt-get install -y --no-install-recommends apt-utils dialog wget ca-certificates \
  #
  # Verify git, common tools / libs installed, add/modify non-root user, optionally install zsh
  && wget -q -O /tmp/common-setup.sh $COMMON_SCRIPT_SOURCE \
  && if [ "$COMMON_SCRIPT_SHA" != "dev-mode" ]; then echo "$COMMON_SCRIPT_SHA /tmp/common-setup.sh" | sha256sum -c - ; fi \
  && /bin/bash /tmp/common-setup.sh "$INSTALL_ZSH" "$USERNAME" "$USER_UID" "$USER_GID" \
  && rm /tmp/common-setup.sh \
  && apt-get clean -y \
  && rm -rf /var/lib/apt/lists/*

RUN install -d -m 770 -o vscode -g vscode /home/vscode/.ssh/ && ssh-keyscan -t rsa github.com >> /home/vscode/.ssh/known_hosts

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=dialog