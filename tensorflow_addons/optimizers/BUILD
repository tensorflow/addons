licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "optimizers",
    srcs = [
        "__init__.py",
        "conditional_gradient.py",
        "lamb.py",
        "lazy_adam.py",
        "lookahead.py",
        "moving_average.py",
        "proximal_adagrad.py",
        "rectified_adam.py",
        "weight_decay_optimizers.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_addons/utils",
    ],
)

py_test(
    name = "lamb_test",
    size = "small",
    srcs = [
        "lamb_test.py",
    ],
    main = "lamb_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":optimizers",
    ],
)

py_test(
    name = "conditional_gradient_test",
    size = "small",
    srcs = [
        "conditional_gradient_test.py",
    ],
    main = "conditional_gradient_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":optimizers",
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
    name = "proximal_adagrad_test",
    size = "small",
    srcs = [
        "proximal_adagrad_test.py",
    ],
    main = "proximal_adagrad_test.py",
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
