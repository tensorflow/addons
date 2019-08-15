licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "activations",
    srcs = [
        "__init__.py",
        "sparsemax.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_addons/utils",
    ],
)

py_test(
    name = "sparsemax_test",
    size = "medium",
    srcs = [
        "sparsemax_test.py",
    ],
    main = "sparsemax_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":activations",
    ],
)

py_test(
    name = "gelu_test",
    size = "medium",
    srcs = [
        "gelu_test.py",
    ],
    main = "gelu_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":activations",
    ],
)
