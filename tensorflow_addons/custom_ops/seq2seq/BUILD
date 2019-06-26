licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("@local_config_tf//:build_defs.bzl", "D_GLIBCXX_USE_CXX11_ABI")

cc_binary(
    name = "_beam_search_ops.so",
    srcs = [
        "cc/kernels/beam_search_ops.cc",
        "cc/kernels/beam_search_ops.h",
        #        "cc/kernels/beam_search_ops_gpu.cu.cc",
        "cc/ops/beam_search_ops.cc",
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
