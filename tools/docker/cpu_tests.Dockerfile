FROM gcr.io/tensorflow-testing/nosla-ubuntu16.04-manylinux2010

COPY build_deps/build-requirements.txt ./
RUN pip3 install -r build-requirements.txt

COPY requirements.txt ./
RUN pip3 install -r requirements.txt

COPY ./ /addons
WORKDIR addons
RUN bash tools/ci_testing/addons_cpu.sh
