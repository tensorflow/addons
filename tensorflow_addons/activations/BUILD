licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "activations_py",
    srcs = [
        "__init__.py",
        "python/__init__.py",
        "python/sparsemax.py"
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_addons/utils:utils_py",
    ],
)

py_test(
    name = "sparsemax_py_test",
    size = "medium",
    srcs = [
        "python/sparsemax_test.py",
    ],
    main = "python/sparsemax_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":activations_py",
    ],
)
