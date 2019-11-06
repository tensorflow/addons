# Description:
#   Doc generator

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(
    default_visibility = ["//tensorflow_addons:__subpackages__"],
)

py_binary(
    name = "build_docs",
    srcs = ["build_docs.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_addons",
    ],
)

py_test(
    name = "tfa_doctest",
    srcs = ["tfa_doctest.py"],
    python_version = "PY3",
    tags = [
        "no_oss_py2",
        "noasan",
        "nomsan",
        "notsan",
    ],
    deps = [
        "//tensorflow_addons",
    ],
)
