licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "layers",
    srcs = [
        "__init__.py",
        "gelu.py",
        "maxout.py",
        "normalizations.py",
        "optical_flow.py",
        "poincare.py",
        "sparsemax.py",
        "tlu.py",
        "wrappers.py",
    ],
    data = [
        "//tensorflow_addons/custom_ops/layers:_correlation_cost_ops.so",
    ],
    deps = [
        "//tensorflow_addons/activations",
        "//tensorflow_addons/utils",
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
        ":layers",
    ],
)

py_test(
    name = "tlu_test",
    size = "small",
    srcs = [
        "tlu_test.py",
    ],
    main = "tlu_test.py",
    deps = [
        ":layers",
    ],
)

py_test(
    name = "layers_wrappers_test",
    size = "small",
    srcs = [
        "wrappers_test.py",
    ],
    main = "wrappers_test.py",
    deps = [
        ":layers",
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
        ":layers",
    ],
)

py_test(
    name = "maxout_test",
    size = "small",
    srcs = [
        "maxout_test.py",
    ],
    main = "maxout_test.py",
    deps = [
        ":layers",
    ],
)

py_test(
    name = "normalizations_test",
    size = "small",
    srcs = [
        "normalizations_test.py",
    ],
    main = "normalizations_test.py",
    deps = [
        ":layers",
    ],
)

py_test(
    name = "optical_flow_test",
    size = "small",
    srcs = [
        "optical_flow_test.py",
    ],
    main = "optical_flow_test.py",
    deps = [
        ":layers",
    ],
)
