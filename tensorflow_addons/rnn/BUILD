licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "rnn",
    srcs = [
        "__init__.py",
        "cell.py",
    ],
    deps = [
        "//tensorflow_addons/utils",
    ],
)

py_test(
    name = "rnn_test",
    size = "small",
    srcs = glob(["*_test.py"]),
    main = "run_all_test.py",
    deps = [
        ":rnn",
    ],
)
