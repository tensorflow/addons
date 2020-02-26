FROM tensorflow/tensorflow:2.1.0-custom-op-gpu-ubuntu16

COPY build_deps/build-requirements.txt ./

RUN python3 -m pip install -r build-requirements.txt

COPY requirements.txt ./
RUN python3 -m pip install -r requirements.txt

COPY tools/docker/finish_bazel_install.sh ./
RUN bash finish_bazel_install.sh


COPY ./ /addons
WORKDIR addons
CMD ["bash", "tools/ci_testing/addons_gpu.sh"]
