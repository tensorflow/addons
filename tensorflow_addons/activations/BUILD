licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "activations",
    srcs = [
        "__init__.py",
        "gelu.py",
        "hardshrink.py",
        "lisht.py",
        "mish.py",
        "rrelu.py",
        "softshrink.py",
        "sparsemax.py",
        "tanhshrink.py",
    ],
    data = [
        "//tensorflow_addons/custom_ops/activations:_activation_ops.so",
        "//tensorflow_addons/utils",
    ],
)

py_test(
    name = "activations_test",
    size = "small",
    srcs = [
        "activations_test.py",
    ],
    main = "activations_test.py",
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
    deps = [
        ":activations",
    ],
)

py_test(
    name = "mish_test",
    size = "small",
    srcs = [
        "mish_test.py",
    ],
    main = "mish_test.py",
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
    deps = [
        ":activations",
    ],
)

py_test(
    name = "rrelu_test",
    size = "small",
    srcs = [
        "rrelu_test.py",
    ],
    main = "rrelu_test.py",
    deps = [
        ":activations",
    ],
)

py_test(
    name = "rrelu_benchmark",
    size = "small",
    srcs = [
        "rrelu_benchmark.py",
    ],
    main = "rrelu_benchmark.py",
    deps = [
        ":activations",
    ],
)
