FROM python:3.6

RUN pip install black==19.10b0 flake8==3.7.9

COPY tools/ci_build/install/buildifier.sh ./buildifier.sh
RUN bash buildifier.sh

COPY tools/ci_build/install/clang-format.sh ./clang-format.sh
RUN bash clang-format.sh

WORKDIR /addons


CMD ["python", "tools/format.py"]
