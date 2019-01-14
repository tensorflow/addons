licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "opt_py",
    srcs = [
        "__init__.py",
        "python/__init__.py",
        "python/lazy_adam_optimizer.py",
    ],
    srcs_version = "PY2AND3",
)


py_test(
    name = "lazy_adam_optimizer_test",
    srcs = [
        "python/lazy_adam_optimizer_test.py"
    ],
    main = "python/lazy_adam_optimizer_test.py",
    deps = [
        ":opt_py",
    ],
    srcs_version = "PY2AND3",
)
