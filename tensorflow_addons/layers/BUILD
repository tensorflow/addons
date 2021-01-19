package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

py_library(
    name = "layers",
    srcs = glob(["*.py"]),
    data = [
        "//tensorflow_addons/custom_ops/layers:_correlation_cost_ops.so",
        "//tensorflow_addons/custom_ops/layers:_embedding_bag_ops.so",
    ],
    deps = [
        "//tensorflow_addons/activations",
        "//tensorflow_addons/rnn",
        "//tensorflow_addons/testing",
        "//tensorflow_addons/text",
        "//tensorflow_addons/utils",
    ],
)

py_test(
    name = "layers_test",
    size = "large",
    srcs = glob(["tests/*"]),
    main = "tests/run_all_test.py",
    deps = [
        ":layers",
    ],
)
