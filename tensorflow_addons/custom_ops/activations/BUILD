licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("@local_config_tf//:build_defs.bzl", "D_GLIBCXX_USE_CXX11_ABI")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda_is_configured", "if_cuda")

cc_library(
    name = "gelu_op_gpu",
    srcs = [
        "cc/kernels/gelu_op.h",
        "cc/kernels/gelu_op_gpu.cu.cc",
    ],
    copts = if_cuda_is_configured([
        "-DGOOGLE_CUDA=1",
        "-x cuda",
        "-nvcc_options=relaxed-constexpr",
        "-nvcc_options=ftz=true",
    ]),
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ] + if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_libs",
        "@local_config_cuda//cuda:cuda_headers",
    ]),
    alwayslink = 1,
)

cc_library(
    name = "hardshrink_op_gpu",
    srcs = [
        "cc/kernels/hardshrink_op.h",
        "cc/kernels/hardshrink_op_gpu.cu.cc",
    ],
    copts = if_cuda_is_configured([
        "-DGOOGLE_CUDA=1",
        "-x cuda",
        "-nvcc_options=relaxed-constexpr",
        "-nvcc_options=ftz=true",
    ]),
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ] + if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_libs",
        "@local_config_cuda//cuda:cuda_headers",
    ]),
    alwayslink = 1,
)

cc_library(
    name = "tanhshrink_op_gpu",
    srcs = [
        "cc/kernels/tanhshrink_op.h",
        "cc/kernels/tanhshrink_op_gpu.cu.cc",
    ],
    copts = if_cuda_is_configured([
        "-DGOOGLE_CUDA=1",
        "-x cuda",
        "-nvcc_options=relaxed-constexpr",
        "-nvcc_options=ftz=true",
    ]),
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ] + if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_libs",
        "@local_config_cuda//cuda:cuda_headers",
    ]),
    alwayslink = 1,
)

cc_binary(
    name = "_activation_ops.so",
    srcs = [
        "cc/kernels/gelu_op.cc",
        "cc/kernels/gelu_op.h",
        "cc/kernels/hardshrink_op.cc",
        "cc/kernels/hardshrink_op.h",
        "cc/kernels/tanhshrink_op.cc",
        "cc/kernels/tanhshrink_op.h",
        "cc/ops/gelu_op.cc",
        "cc/ops/hardshrink_op.cc",
        "cc/ops/tanhshrink_op.cc",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
        D_GLIBCXX_USE_CXX11_ABI,
    ] + if_cuda(["-DGOOGLE_CUDA=1"]),
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ] + if_cuda_is_configured([
        ":gelu_op_gpu",
        ":hardshrink_op_gpu",
        ":tanhshrink_op_gpu",
    ]),
)
