FROM tensorflow/tensorflow:2.1.0-custom-op-gpu-ubuntu16

COPY build_deps/build-requirements.txt ./
RUN pip3 install -r build-requirements.txt

COPY requirements.txt ./
RUN pip3 install -r requirements.txt

COPY ./ /addons
WORKDIR addons
CMD ["bash", "tools/ci_testing/addons_gpu.sh"]
