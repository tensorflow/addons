licenses(["notice"])  # Apache 2.0

load("//third_party/tensorflow:tensorflow.google.bzl", "tf_custom_op_py_library")

package(default_visibility = ["//visibility:public"])

tf_custom_op_py_library(
    name = "layers",
    srcs = [
        "__init__.py",
        "gelu.py",
        "maxout.py",
        "netvlad.py",
        "normalizations.py",
        "optical_flow.py",
        "poincare.py",
        "polynomial.py",
        "sparsemax.py",
        "tlu.py",
        "wrappers.py",
    ],
    dso = [
        "//third_party/py/tensorflow_addons/custom_ops/layers/cc:_correlation_cost_ops.so",
    ],
    kernels = [
        "//third_party/py/tensorflow_addons/custom_ops/layers/cc:correlation_cost_ops_kernels",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_addons/activations",
        "//third_party/py/tensorflow_addons/utils",
    ],
)

py_test(
    name = "polynomial_test",
    size = "small",
    srcs = [
        "polynomial_test.py",
    ],
    main = "polynomial_test.py",
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":layers",
    ],
)

py_test(
    name = "gelu_test",
    size = "small",
    srcs = [
        "gelu_test.py",
    ],
    main = "gelu_test.py",
    python_version = "PY3",
    srcs_version = "PY2AND3",
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
    python_version = "PY3",
    srcs_version = "PY2AND3",
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
    python_version = "PY3",
    srcs_version = "PY2AND3",
    # TODO: Investigate why flaky #830
    tags = ["notap"],
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
    python_version = "PY3",
    srcs_version = "PY2AND3",
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
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":layers",
    ],
)

py_test(
    name = "normalizations_test",
    size = "medium",
    srcs = [
        "normalizations_test.py",
    ],
    main = "normalizations_test.py",
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":layers",
    ],
)

py_test(
    name = "optical_flow_test",
    size = "medium",
    srcs = [
        "optical_flow_test.py",
    ],
    main = "optical_flow_test.py",
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":layers",
    ],
)

py_test(
    name = "netvlad_test",
    size = "medium",
    srcs = [
        "netvlad_test.py",
    ],
    main = "netvlad_test.py",
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":layers",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_addons/utils",
    ],
)
