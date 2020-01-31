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
        "discriminative_layer_training.py"
    ],
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
    deps = [
        ":optimizers",
    ],
)

py_test(
    name = "yogi_test",
    size = "small",
    srcs = [
        "yogi_test.py",
    ],
    main = "yogi_test.py",
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
    deps = [
        ":optimizers",
    ],
)

py_test(
    name = "stochastic_weight_averaging_test",
    size = "small",
    srcs = [
        "stochastic_weight_averaging_test.py",
    ],
    main = "stochastic_weight_averaging_test.py",
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
    deps = [
        ":optimizers",
    ],
)

py_test(
    name = "novograd_test",
    size = "small",
    srcs = [
        "novograd_test.py",
    ],
    main = "novograd_test.py",
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



py_test(
    name = "discriminative_layer_training_test",
    size = "medium",
    srcs = [
        "discriminative_layer_training_test.py",
    ],
    main = "discriminative_layer_training_test.py",
    deps = [
        ":optimizers",
    ],
)
