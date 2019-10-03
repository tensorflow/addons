licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("@local_config_tf//:build_defs.bzl", "D_GLIBCXX_USE_CXX11_ABI")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda_is_configured", "if_cuda")

cc_binary(
    name = "_correlation_cost_ops.so",
    srcs = [
        "cc/kernels/correlation_cost_op.cc",
        "cc/kernels/correlation_cost_op.h",
        "cc/ops/correlation_cost_op.cc",
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
    ] + if_cuda_is_configured([":correlation_cost_ops_gpu"]),
)

cc_library(
    name = "correlation_cost_ops_gpu",
    srcs = [
        "cc/kernels/correlation_cost_op.h",
        "cc/kernels/correlation_cost_op_gpu.cu.cc",
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
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart_static",
        "@cub_archive//:cub",
    ]),
    alwayslink = 1,
)
