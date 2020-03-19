licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("//tensorflow_addons:tensorflow_addons.bzl", "custom_op_library")

custom_op_library(
    name = "_optimizer_ops.so",
    srcs = [
        "cc/kernels/adam_weightdecay_op.cc",
        "cc/kernels/adam_weightdecay_op.h",
        "cc/ops/adam_weightdecay_op.cc",
    ],
    cuda_srcs = [
        "cc/kernels/adam_weightdecay_op.h",
        "cc/kernels/adam_weightdecay_op_gpu.cu.cc",
    ],
)
