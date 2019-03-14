licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])


py_library(
    name = "images_ops_py",
    srcs = ([
        "__init__.py",
        "distort_image_ops.py",
        "transform_ops.py",
    ]),
    data = [
        "//tensorflow_addons/custom_ops/image:_distort_image_ops.so",
        "//tensorflow_addons/custom_ops/image:_image_ops.so",
        "//tensorflow_addons/utils:utils_py",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "distort_image_ops_test",
    size = "small",
    srcs = [
        "distort_image_ops_test.py",
    ],
    main = "distort_image_ops_test.py",
    deps = [
        ":images_ops_py",
    ],
    srcs_version = "PY2AND3"
)

# TODO: use cuda_py_test later.
py_test(
    name = "transform_ops_test",
    size = "medium",
    srcs = [
        "transform_test.py",
    ],
    main = "transform_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":images_ops_py",
    ],
)
