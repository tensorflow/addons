licenses(["notice"])  # Apache 2.0

load(
    "//third_party/tensorflow:tensorflow.bzl",
    "tf_custom_op_library",
    "tf_gen_op_libs",
    "tf_gen_op_wrapper_py",
    "tf_kernel_library",
)

package(default_visibility = ["//visibility:public"])

tf_custom_op_library(
    name = "_sparse_ops.so",
    srcs = [
        "kernels/sparse_bicgstab_solver.cc",
    ],
)

tf_kernel_library(
    name = "sparse_ops_kernels",
    srcs = [
        "kernels/sparse_bicgstab_solver.cc",
    ],
    deps = [
        ":sparse_ops_op_lib",
        "//third_party/eigen3",
        "//third_party/tensorflow/core:framework",
        "//third_party/tensorflow/core:framework_lite",
        "//third_party/tensorflow/core/kernels/sparse:kernels",
        "//third_party/tensorflow/core/kernels/sparse:sparse_matrix",
    ],
    alwayslink = 1,
)

tf_gen_op_libs(
    op_lib_names = ["sparse_ops"],
)

tf_gen_op_wrapper_py(
    name = "sparse_ops",
    deps = [":sparse_ops_op_lib"],
)
