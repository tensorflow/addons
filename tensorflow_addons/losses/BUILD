licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "losses",
    srcs = glob(["*.py"]),
    deps = [
        "//tensorflow_addons/activations",
        "//tensorflow_addons/testing",
        "//tensorflow_addons/utils",
    ],
)

py_test(
    name = "losses_test",
    size = "small",
    srcs = glob(["tests/*"]),
    main = "tests/run_all_test.py",
    deps = [
        ":losses",
    ],
)
