licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "activations",
    srcs = [
        "__init__.py",
        "gelu.py",
        "hardshrink.py",
        "rrelu.py",
        "sparsemax.py",
        "tanhshrink.py",
    ],
    data = [
        "//tensorflow_addons/custom_ops/activations:_activation_ops.so",
        "//tensorflow_addons/utils",
    ],
    srcs_version = "PY2AND3",
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

py_test(
    name = "hardshrink_test",
    size = "medium",
    srcs = [
        "hardshrink_test.py",
    ],
    main = "hardshrink_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":activations",
    ],
)

py_test(
    name = "tanhshrink_test",
    size = "medium",
    srcs = [
        "tanhshrink_test.py",
    ],
    main = "tanhshrink_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":activations",
    ],
)

py_test(
    name = "rrelu_test",
    size = "medium",
    srcs = [
        "rrelu_test.py",
    ],
    main = "rrelu_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":activations",
    ],
)
