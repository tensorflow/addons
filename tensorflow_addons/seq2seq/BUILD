licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "python/_beam_search_ops.so",
    srcs = [
        "cc/kernels/beam_search_ops.cc",
        "cc/kernels/beam_search_ops.h",
        #        "cc/kernels/beam_search_ops_gpu.cu.cc",
        "cc/ops/beam_search_ops.cc",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

py_library(
    name = "seq2seq",
    srcs = [
        "__init__.py",
        "python/attention_wrapper.py",
        "python/basic_decoder.py",
        "python/beam_search_decoder.py",
        "python/decoder.py",
        "python/loss.py",
        "python/sampler.py",
    ],
    data = [
        ":python/_beam_search_ops.so",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_addons/utils:utils_py",
    ],
)

py_test(
    name = "attention_wrapper_test",
    size = "medium",
    srcs = ["python/attention_wrapper_test.py"],
    deps = [
        ":seq2seq",
    ],
)

py_test(
    name = "basic_decoder_test",
    size = "medium",
    srcs = ["python/basic_decoder_test.py"],
    deps = [
        ":seq2seq",
    ],
)

py_test(
    name = "beam_search_decoder_test",
    size = "medium",
    srcs = ["python/beam_search_decoder_test.py"],
    deps = [
        ":seq2seq",
    ],
)

py_test(
    name = "beam_search_ops_test",
    size = "medium",
    srcs = ["python/beam_search_ops_test.py"],
    deps = [
        ":seq2seq",
    ],
)

py_test(
    name = "decoder_test",
    size = "medium",
    srcs = ["python/decoder_test.py"],
    deps = [
        ":seq2seq",
    ],
)

py_test(
    name = "loss_test",
    size = "medium",
    srcs = ["python/loss_test.py"],
    deps = [
        ":seq2seq",
    ],
)
