licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "seq2seq",
    srcs = glob(["*.py"]),
    data = [
        "//tensorflow_addons:options.py",
        "//tensorflow_addons/custom_ops/seq2seq:_beam_search_ops.so",
    ],
    deps = [
        "//tensorflow_addons/testing",
        "//tensorflow_addons/utils",
    ],
)

py_test(
    name = "seq2seq_test",
    size = "medium",
    srcs = glob(["tests/*"]),
    main = "tests/run_all_test.py",
    deps = [
        ":seq2seq",
    ],
)
