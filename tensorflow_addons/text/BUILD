licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])


py_library(
    name = "text_ops",
    srcs = ([
        "skip_gram_ops.py",
    ]),
    data = [
        "//tensorflow_addons/custom_ops/text:_skip_gram_ops.so",
        "//tensorflow_addons/utils:utils",
    ],
    srcs_version = "PY2AND3",
)

py_library(
    name = "text",
    srcs = ([
        "__init__.py",
    ]),
    srcs_version = "PY2AND3",
    deps = [
        ":text_ops",
    ],
)

py_test(
    name = "text_ops_test",
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
