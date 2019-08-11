licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("@local_config_tf//:build_defs.bzl", "D_GLIBCXX_USE_CXX11_ABI")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda_is_configured", "if_cuda")

cc_library(
    name = "distort_image_ops_gpu",
    srcs = [
        "cc/kernels/adjust_hsv_in_yiq_op.h",
        "cc/kernels/adjust_hsv_in_yiq_op_gpu.cu.cc",
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
    name = "_distort_image_ops.so",
    srcs = [
        "cc/kernels/adjust_hsv_in_yiq_op.cc",
        "cc/kernels/adjust_hsv_in_yiq_op.h",
        "cc/ops/distort_image_ops.cc",
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
    ] + if_cuda_is_configured([":distort_image_ops_gpu"]),
)

cc_library(
    name = "image_projective_transform_op_gpu",
    srcs = [
        "cc/kernels/image_projective_transform_op.h",
        "cc/kernels/image_projective_transform_op_gpu.cu.cc",
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
    name = "euclidean_distance_transform_op_gpu",
    srcs = [
        "cc/kernels/euclidean_distance_transform_op.h",
        "cc/kernels/euclidean_distance_transform_op_gpu.cu.cc",
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
    name = "_image_ops.so",
    srcs = [
        "cc/kernels/euclidean_distance_transform_op.cc",
        "cc/kernels/euclidean_distance_transform_op.h",
        "cc/kernels/image_projective_transform_op.cc",
        "cc/kernels/image_projective_transform_op.h",
        "cc/ops/image_ops.cc",
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
        ":image_projective_transform_op_gpu",
        ":euclidean_distance_transform_op_gpu",
    ]),
)
