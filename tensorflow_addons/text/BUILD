licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])


cc_binary(
    name = 'python/ops/_skip_gram_ops.so',
    srcs = [
        "cc/kernels/skip_gram_kernels.cc",
        "cc/ops/skip_gram_ops.cc",
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    copts = ["-pthread", "-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0"]
)


py_library(
    name = "text_ops_py",
    srcs = ([
        "python/ops/skip_gram_ops.py",
    ]),
    data = [
        ":python/ops/_skip_gram_ops.so"
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "text_ops_py_test",
    srcs = [
        "python/ops/skip_gram_ops_test.py"
    ],
    main = "python/ops/skip_gram_ops_test.py",
    deps = [
        ":text_ops_py",
    ],
    srcs_version = "PY2AND3",
)

py_library(
    name = "text_py",
    srcs = ([
        "__init__.py",
        "python/__init__.py",
        "python/ops/__init__.py",
    ]),
    deps = [
        ":text_ops_py"
    ],
    srcs_version = "PY2AND3",
)