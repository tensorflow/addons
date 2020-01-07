licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("//tensorflow_addons:tensorflow_addons.bzl", "custom_op_library")

custom_op_library(
    name = "_activation_ops.so",
    srcs = [
        "cc/kernels/gelu_op.cc",
        "cc/kernels/gelu_op.h",
        "cc/kernels/hardshrink_op.cc",
        "cc/kernels/hardshrink_op.h",
        "cc/kernels/lisht_op.cc",
        "cc/kernels/lisht_op.h",
        "cc/kernels/mish_op.cc",
        "cc/kernels/mish_op.h",
        "cc/kernels/softshrink_op.cc",
        "cc/kernels/softshrink_op.h",
        "cc/kernels/tanhshrink_op.cc",
        "cc/kernels/tanhshrink_op.h",
        "cc/ops/gelu_op.cc",
        "cc/ops/hardshrink_op.cc",
        "cc/ops/lisht_op.cc",
        "cc/ops/mish_op.cc",
        "cc/ops/softshrink_op.cc",
        "cc/ops/tanhshrink_op.cc",
    ],
    cuda_srcs = [
        "cc/kernels/gelu_op.h",
        "cc/kernels/gelu_op_gpu.cu.cc",
        "cc/kernels/hardshrink_op.h",
        "cc/kernels/hardshrink_op_gpu.cu.cc",
        "cc/kernels/lisht_op.h",
        "cc/kernels/lisht_op_gpu.cu.cc",
        "cc/kernels/mish_op.h",
        "cc/kernels/mish_op_gpu.cu.cc",
        "cc/kernels/softshrink_op.h",
        "cc/kernels/softshrink_op_gpu.cu.cc",
        "cc/kernels/tanhshrink_op.h",
        "cc/kernels/tanhshrink_op_gpu.cu.cc",
    ],
)
