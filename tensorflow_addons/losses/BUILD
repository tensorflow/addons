licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "losses_py",
    srcs = [
        "__init__.py",
        "python/__init__.py",
        "python/triplet.py",
        "python/metric_learning.py",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "triplet_py_test",
    size = "small",
    srcs = [
        "python/triplet_test.py",
    ],
    main = "python/triplet_test.py",
    deps = [
        ":losses_py",
    ],
    srcs_version = "PY2AND3",
)
