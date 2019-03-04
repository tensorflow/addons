licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "losses_py",
    srcs = [
        "__init__.py",
        "python/__init__.py",
        "python/lifted.py",
        "python/metric_learning.py",
        "python/triplet.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_addons/utils:utils_py",
    ],
)

py_test(
    name = "triplet_py_test",
    size = "small",
    srcs = [
        "python/triplet_test.py",
    ],
    main = "python/triplet_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":losses_py",
    ],
)

py_test(
    name = "lifted_py_test",
    size = "small",
    srcs = [
        "python/lifted_test.py",
    ],
    main = "python/lifted_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":losses_py",
    ],
)
