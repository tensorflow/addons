licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("//tensorflow_addons:tensorflow_addons.bzl", "custom_op_library")

custom_op_library(
    name = "_beam_search_ops.so",
    srcs = [
        "cc/kernels/beam_search_ops.cc",
        "cc/kernels/beam_search_ops.h",
        "cc/ops/beam_search_ops.cc",
    ],
    cuda_srcs = [
        "cc/kernels/beam_search_ops.h",
        "cc/kernels/beam_search_ops_gpu.cu.cc",
    ],
)
