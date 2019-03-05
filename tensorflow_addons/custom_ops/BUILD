licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "custom_ops_py",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_addons/custom_ops/image:images_ops_py",
        "//tensorflow_addons/custom_ops/text:text_py",
    ],
)
