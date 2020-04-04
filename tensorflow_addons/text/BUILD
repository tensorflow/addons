licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

# TODO: Once TF exports symbols in a DLL we can enable parse_time_op for windows
# https://github.com/tensorflow/addons/issues/782
py_library(
    name = "text",
    srcs = [
        "__init__.py",
        "crf.py",
        "parse_time_op.py",
        "skip_gram_ops.py",
    ],
    data = select({
        "//tensorflow_addons:windows": [
            "//tensorflow_addons/custom_ops/text:_skip_gram_ops.so",
            "//tensorflow_addons/utils",
        ],
        "//conditions:default": [
            "//tensorflow_addons/custom_ops/text:_parse_time_op.so",
            "//tensorflow_addons/custom_ops/text:_skip_gram_ops.so",
            "//tensorflow_addons/utils",
        ],
    }),
)

py_test(
    name = "text_test",
    size = "small",
    srcs = glob(["*_test.py"]),
    main = "run_all_test.py",
    deps = [
        ":text",
    ],
)
