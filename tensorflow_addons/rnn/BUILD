licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "rnn",
    srcs = [
        "__init__.py",
        "cell.py",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "cell_test",
    size = "medium",
    srcs = ["cell_test.py"],
    deps = [
        ":rnn",
        "//tensorflow_addons/utils",
    ],
)
