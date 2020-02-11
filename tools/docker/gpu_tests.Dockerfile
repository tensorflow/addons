FROM gcr.io/tensorflow-testing/nosla-cuda10.1-cudnn7-ubuntu16.04-manylinux2010

COPY build_deps/build-requirements.txt ./
RUN pip3 install -r build-requirements.txt

COPY requirements.txt ./
RUN pip3 install -r --no-cache-dir requirements.txt

COPY ./ /addons
WORKDIR addons
CMD ["bash", "tools/ci_testing/addons_gpu.sh"]
