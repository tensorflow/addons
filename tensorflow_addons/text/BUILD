licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

# TODO: Once TF exports symbols in a DLL we can enable parse_time_op for windows
py_library(
    name = "text",
    srcs = select({
        "//tensorflow_addons:windows": ["__init__.py",
                                        "crf.py",
                                        "skip_gram_ops.py",],
        "//conditions:default": ["__init__.py",
                                 "crf.py",
                                 "parse_time_op.py",
                                 "skip_gram_ops.py",],
    }),
    data = select({
        "//tensorflow_addons:windows": ["//tensorflow_addons/custom_ops/text:_skip_gram_ops.so",
                                        "//tensorflow_addons/utils"],

        "//conditions:default": ["//tensorflow_addons/custom_ops/text:_parse_time_op.so",
                                 "//tensorflow_addons/custom_ops/text:_skip_gram_ops.so",
                                 "//tensorflow_addons/utils"],
    }),
)

py_test(
    name = "crf_test",
    size = "small",
    srcs = [
        "crf_test.py",
    ],
    main = "crf_test.py",
    deps = [
        ":text",
    ],
)

py_test(
    name = "skip_gram_ops_test",
    size = "small",
    srcs = [
        "skip_gram_ops_test.py",
    ],
    main = "skip_gram_ops_test.py",
    deps = [
        ":text",
    ],
)

# Temporarily use dummy test for windows ParseTime
py_test(
    name = "parse_time_op_test",
    size = "small",
    srcs = select({
        "//tensorflow_addons:windows": ["parse_time_dummy_test.py"],
        "//conditions:default": ["parse_time_op_test.py"],
    }),

    main = select({
        "//tensorflow_addons:windows": "parse_time_dummy_test.py",
        "//conditions:default": "parse_time_op_test.py",
    }),
    deps = [
        ":text",
    ],
)
