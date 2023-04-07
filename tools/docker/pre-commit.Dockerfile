FROM python:3.9

COPY tools/install_deps /install_deps
RUN pip install -r /install_deps/black.txt -r /install_deps/flake8.txt

COPY tools/install_deps/buildifier.sh ./buildifier.sh
RUN bash buildifier.sh

COPY tools/install_deps/clang-format.sh ./clang-format.sh
RUN bash clang-format.sh

WORKDIR /addons


CMD ["python", "tools/format.py"]
