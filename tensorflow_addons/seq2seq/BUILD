licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "seq2seq",
    srcs = [
        "__init__.py",
        "attention_wrapper.py",
        "basic_decoder.py",
        "beam_search_decoder.py",
        "decoder.py",
        "loss.py",
        "sampler.py",
    ],
    data = [
        "//tensorflow_addons/custom_ops/seq2seq:_beam_search_ops.so",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_addons/utils",
    ],
)

py_test(
    name = "attention_wrapper_test",
    size = "medium",
    srcs = ["attention_wrapper_test.py"],
    deps = [
        ":seq2seq",
    ],
)

py_test(
    name = "basic_decoder_test",
    size = "small",
    srcs = ["basic_decoder_test.py"],
    deps = [
        ":seq2seq",
    ],
)

py_test(
    name = "beam_search_decoder_test",
    size = "small",
    srcs = ["beam_search_decoder_test.py"],
    deps = [
        ":seq2seq",
    ],
)

py_test(
    name = "beam_search_ops_test",
    size = "small",
    srcs = ["beam_search_ops_test.py"],
    deps = [
        ":seq2seq",
    ],
)

py_test(
    name = "decoder_test",
    size = "small",
    srcs = ["decoder_test.py"],
    deps = [
        ":seq2seq",
    ],
)

py_test(
    name = "loss_test",
    size = "small",
    srcs = ["loss_test.py"],
    deps = [
        ":seq2seq",
    ],
)
