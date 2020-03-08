FROM python:3.6

COPY tools/install_deps /install_deps

RUN pip install -r /install_deps/black.txt -r /install_deps/flake8.txt
RUN bash /install_deps/buildifier.sh
RUN bash /install_deps/clang-format.sh

WORKDIR /addons


CMD ["python", "tools/format.py"]
