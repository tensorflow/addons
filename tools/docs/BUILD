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
    deps = [
        "//tensorflow_addons",
    ],
)
