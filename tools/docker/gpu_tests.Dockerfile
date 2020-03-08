FROM tensorflow/tensorflow:2.1.0-custom-op-gpu-ubuntu16

COPY tools/install_deps /install_deps

RUN python3 -m pip install -r /install_deps/tensorflow.txt
RUN bash /install_deps/finish_bazel_install.sh

COPY requirements.txt ./
RUN python3 -m pip install -r requirements.txt

COPY ./ /addons
WORKDIR addons
CMD ["bash", "tools/ci_testing/addons_gpu.sh"]
