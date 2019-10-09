licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "activations",
    srcs = [
        "__init__.py",
        "gelu.py",
        "hardshrink.py",
        "lisht.py",
        "softshrink.py",
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
    name = "activations_test",
    size = "small",
    srcs = [
        "activations_test.py",
    ],
    main = "activations_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":activations",
    ],
)

py_test(
    name = "sparsemax_test",
    size = "small",
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
    size = "small",
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
    size = "small",
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
    name = "lisht_test",
    size = "small",
    srcs = [
        "lisht_test.py",
    ],
    main = "lisht_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":activations",
    ],
)

py_test(
    name = "softshrink_test",
    size = "small",
    srcs = [
        "softshrink_test.py",
    ],
    main = "softshrink_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":activations",
    ],
)

py_test(
    name = "tanhshrink_test",
    size = "small",
    srcs = [
        "tanhshrink_test.py",
    ],
    main = "tanhshrink_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":activations",
    ],
)
