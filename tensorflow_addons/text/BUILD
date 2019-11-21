licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

# TODO: https://github.com/tensorflow/addons/issues/663
py_library(
    name = "text",
    srcs = ([
        "__init__.py",
        "crf.py",
        # "parse_time_op.py",
        "skip_gram_ops.py",
    ]),
    data = [
        # "//tensorflow_addons/custom_ops/text:_parse_time_op.so",
        "//tensorflow_addons/custom_ops/text:_skip_gram_ops.so",
        "//tensorflow_addons/utils",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "crf_test",
    size = "small",
    srcs = [
        "crf_test.py",
    ],
    main = "crf_test.py",
    srcs_version = "PY2AND3",
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
    srcs_version = "PY2AND3",
    deps = [
        ":text",
    ],
)

# TODO: https://github.com/tensorflow/addons/issues/663
# py_test(
#     name = "parse_time_op_test",
#     size = "small",
#     srcs = [
#         "parse_time_op_test.py",
#     ],
#     main = "parse_time_op_test.py",
#     srcs_version = "PY2AND3",
#     deps = [
#         ":text",
#     ],
# )
