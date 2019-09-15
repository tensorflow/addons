licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "optimizers",
    srcs = [
        "__init__.py",
        "lazy_adam.py",
        "lookahead.py",
        "moving_average.py",
        "rectified_adam.py",
        "weight_decay_optimizers.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_addons/utils",
    ],
)

py_test(
    name = "lazy_adam_test",
    size = "small",
    srcs = [
        "lazy_adam_test.py",
    ],
    main = "lazy_adam_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":optimizers",
    ],
)

py_test(
    name = "lookahead_test",
    size = "small",
    srcs = [
        "lookahead_test.py",
    ],
    main = "lookahead_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":optimizers",
    ],
)

py_test(
    name = "moving_average_test",
    size = "small",
    srcs = [
        "moving_average_test.py",
    ],
    main = "moving_average_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":optimizers",
    ],
)

py_test(
    name = "rectified_adam_test",
    size = "small",
    srcs = [
        "rectified_adam_test.py",
    ],
    main = "rectified_adam_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":optimizers",
    ],
)

py_test(
    name = "weight_decay_optimizers_test",
    size = "small",
    srcs = [
        "weight_decay_optimizers_test.py",
    ],
    main = "weight_decay_optimizers_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":optimizers",
    ],
)
