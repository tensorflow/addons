licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])


cc_binary(
    name = "python/_image_ops.so",
    srcs = [
        "cc/kernels/image_projective_transform_op.cc",
        "cc/kernels/image_projective_transform_op.h",
        "cc/ops/image_ops.cc",
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    copts = ["-pthread", "-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0"]
)


py_library(
    name = "images_ops_py",
    srcs = ([
        "__init__.py",
        "python/__init__.py",
        "python/transform.py",
    ]),
    data = [
        ":python/_image_ops.so"
    ],
    srcs_version = "PY2AND3",
)


# TODO: use cuda_py_test later.
py_test(
    name = "transform_ops_test",
    size = "small",
    srcs = [
        "python/transform_test.py"
    ],
    main = "python/transform_test.py",
    deps = [
        ":images_ops_py",
    ],
    srcs_version = "PY2AND3",
)

