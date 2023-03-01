licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "windows",
    constraint_values = ["@platforms//os:windows"],
)

py_library(
    name = "tensorflow_addons",
    srcs = glob(["*.py"]),
    deps = [
        "//tensorflow_addons/activations",
        "//tensorflow_addons/callbacks",
        "//tensorflow_addons/image",
        "//tensorflow_addons/layers",
        "//tensorflow_addons/losses",
        "//tensorflow_addons/metrics",
        "//tensorflow_addons/optimizers",
        "//tensorflow_addons/rnn",
        "//tensorflow_addons/seq2seq",
        "//tensorflow_addons/testing",
        "//tensorflow_addons/text",
        "//tensorflow_addons/utils",
    ],
)

py_test(
    name = "tensorflow_addons_test",
    size = "small",
    srcs = glob(["tests/*"]),
    main = "tests/run_all_test.py",
    deps = [
        ":tensorflow_addons",
    ],
)
