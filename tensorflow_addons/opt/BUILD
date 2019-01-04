licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_test(
    name = "opt_py_test",
    srcs = [
        "python/opt/lazy_adam_optimizer_test.py",
    ],
    main = "python/opt/lazy_adam_optimizer_test.py",
    srcs_version = "PY2AND3",
)

py_library(
    name = "opt_py",
    srcs = ([
        "__init__.py",
        "python/__init__.py",
        "python/opt/__init__.py",
        "python/opt/lazy_adam_optimizer.py",
    ]),
    srcs_version = "PY2AND3",
)