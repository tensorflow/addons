licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "opticalflow",
    srcs = ([
        "__init__.py",
        "correlation_cost.py",
    ]),
    data = [
        "//tensorflow_addons/custom_ops/opticalflow:_correlation_cost_ops.so",
        "//tensorflow_addons/utils",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "correlation_cost_test",
    size = "small",
    srcs = [
        "correlation_cost_test.py",
    ],
    main = "correlation_cost_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":opticalflow",
    ],
)
