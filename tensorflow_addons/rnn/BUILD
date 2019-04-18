licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "rnn",
    srcs = [
        "__init__.py",
        "cell.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_addons/utils",
    ],
)

py_test(
    name = "cell_test",
    size = "small",
    srcs = ["cell_test.py"],
    deps = [
        ":rnn",
    ],
)
