licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "python/_distort_image_ops.so",
    srcs = [
        "cc/kernels/adjust_hsv_in_yiq_op.cc",
        "cc/kernels/adjust_hsv_in_yiq_op.h",
        "cc/ops/distort_image_ops.cc",
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    copts = ["-pthread", "-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0"]
)


cc_binary(
    name = "python/_image_ops.so",
    srcs = [
        "cc/kernels/image_projective_transform_op.cc",
        "cc/kernels/image_projective_transform_op.h",
        "cc/ops/image_ops.cc",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

py_library(
    name = "images_ops_py",
    srcs = ([
        "__init__.py",
        "python/__init__.py",
        "python/distort_image_ops.py",
        "python/transform.py",
    ]),
    data = [
        ":python/_distort_image_ops.so",
        ":python/_image_ops.so",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "distort_image_ops_test",
    size = "small",
    srcs = [
        "python/distort_image_ops_test.py",
    ],
    main = "python/distort_image_ops_test.py",
    deps = [
        ":images_ops_py",
    ],
    srcs_version = "PY2AND3"
)

# TODO: use cuda_py_test later.
py_test(
    name = "transform_ops_test",
    size = "small",
    srcs = [
        "python/transform_test.py",
    ],
    main = "python/transform_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":images_ops_py",
    ],
)
