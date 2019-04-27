licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "optimizers",
    srcs = [
        "__init__.py",
        "lazy_adam.py",
        "moving_average.py",
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
