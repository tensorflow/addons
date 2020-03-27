FROM tensorflow/tensorflow:2.1.0-custom-op-gpu-ubuntu16

COPY tools/install_deps/tensorflow.txt ./

RUN python3 -m pip install -r tensorflow.txt

COPY requirements.txt ./
RUN python3 -m pip install -r requirements.txt


COPY ./ /addons
WORKDIR addons
CMD ["bash", "tools/testing/addons_gpu.sh"]
