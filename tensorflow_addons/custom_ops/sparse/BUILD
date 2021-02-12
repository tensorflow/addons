package(default_visibility = ["//visibility:public"])

load("//tensorflow_addons:tensorflow_addons.bzl", "custom_op_library")

custom_op_library(
    name = "_sparse_ops.so",
    srcs = [
        "cc/kernels/sparse_bicgstab_solver.cc",
        "cc/ops/sparse_ops.cc",
    ] + glob([
        "Eigen/IterativeLinearSolvers",
    ]),
)
