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
