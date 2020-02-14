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
    deps = [
        "//tensorflow_addons/utils",
    ],
)
