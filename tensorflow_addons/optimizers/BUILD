licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "optimizers",
    srcs = glob(["*.py"]),
    deps = [
        "//tensorflow_addons/testing",
        "//tensorflow_addons/utils",
    ],
)

py_test(
    name = "optimizers_test",
    size = "small",
    srcs = glob(["tests/*"]),
    main = "tests/run_all_test.py",
    deps = [
        ":optimizers",
    ],
)
