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
    name = "_deformable_conv2d_ops.so",
    srcs = [
        "cc/kernels/deformable_conv2d_op.cc",
        "cc/kernels/deformable_conv2d_op.h",
        "cc/ops/deformable_conv2d_op.cc",
    ],
    cuda_deps = [
        "@cub_archive//:cub",
    ],
    cuda_srcs = [
    ],
)
