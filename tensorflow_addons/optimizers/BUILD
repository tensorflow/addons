licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "optimizers_py",
    srcs = [
        "__init__.py",
        "lazy_adam_optimizer.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_addons/utils:utils_py",
    ],
)

py_test(
    name = "lazy_adam_optimizer_test",
    size = "small",
    srcs = [
        "lazy_adam_optimizer_test.py",
    ],
    main = "lazy_adam_optimizer_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":optimizers_py",
    ],
)
