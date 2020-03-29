licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("//tensorflow_addons:tensorflow_addons.bzl", "custom_op_library")

custom_op_library(
    name = "_skip_gram_ops.so",
    srcs = [
        "cc/kernels/skip_gram_kernels.cc",
        "cc/ops/skip_gram_ops.cc",
    ],
)

custom_op_library(
    name = "_parse_time_op.so",
    srcs = select({
        "//tensorflow_addons:windows": [],
        "//conditions:default": [
            "cc/kernels/parse_time_kernel.cc",
            "cc/ops/parse_time_op.cc",
        ],
    }),
)
