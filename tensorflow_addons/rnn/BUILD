licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "rnn",
    srcs = [
        "__init__.py",
        "cell.py",
        "layernorm_simplernn.py",
    ],
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

py_test(
    name = "layernorm_simplernn_test",
    size = "small",
    srcs = ["layernorm_simplernn_test.py"],
    deps = [
        ":rnn",
    ],
)
