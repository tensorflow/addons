licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("//tensorflow_addons:tensorflow_addons.bzl", "custom_op_library")

custom_op_library(
    name = "_correlation_cost_ops.so",
    srcs = [
        "cc/kernels/correlation_cost_op.cc",
        "cc/kernels/correlation_cost_op.h",
        "cc/ops/correlation_cost_op.cc",
    ],
    cuda_deps = [
        "@cub_archive//:cub",
    ],
    cuda_srcs = [
        "cc/kernels/correlation_cost_op.h",
        "cc/kernels/correlation_cost_op_gpu.cu.cc",
    ],
)

custom_op_library(
    name = "_embeddingbag_ops.so",
    srcs = [
        "cc/kernels/embeddingbag.h",
        "cc/kernels/embeddingbag_backward.h",
        "cc/kernels/embeddingbag_kernels.cc",
        "cc/kernels/embeddingbag_backward_kernels.cc",
        "cc/ops/embeddingbag_ops.cc",
    ],
    cuda_deps = [
        "@cub_archive//:cub",
    ],
    cuda_srcs = [
        "cc/kernels/embeddingbag.h", 
        "cc/kernels/embeddingbag_kernels.cu.cc", 
        "cc/kernels/embeddingbag_backward.h", 
        "cc/kernels/embeddingbag_backward_kernels.cu.cc"
    ],
)
