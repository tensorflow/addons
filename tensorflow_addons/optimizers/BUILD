licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "optimizers",
    srcs = [
        "__init__.py",
        "average_wrapper.py",
        "conditional_gradient.py",
        "cyclical_learning_rate.py",
        "lamb.py",
        "lazy_adam.py",
        "lookahead.py",
        "moving_average.py",
        "novograd.py",
        "rectified_adam.py",
        "stochastic_weight_averaging.py",
        "utils.py",
        "weight_decay_optimizers.py",
        "yogi.py",
    ],
    deps = [
        "//tensorflow_addons/utils",
    ],
)

py_test(
    name = "conditional_gradient_test",
    size = "small",
    srcs = [
        "conditional_gradient_test.py",
    ],
    main = "conditional_gradient_test.py",
    deps = [
        ":optimizers",
    ],
)

py_test(
    name = "cyclical_learning_rate_test",
    size = "small",
    srcs = [
        "cyclical_learning_rate_test.py",
    ],
    main = "cyclical_learning_rate_test.py",
    deps = [
        ":optimizers",
    ],
)
