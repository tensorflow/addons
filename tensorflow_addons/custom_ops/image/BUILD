licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("//tensorflow_addons:addons_bazel_macros.bzl", "D_GLIBCXX_USE_CXX11_ABI")

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
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
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
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)
