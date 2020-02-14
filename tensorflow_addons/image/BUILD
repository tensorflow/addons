licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "image",
    srcs = ([
        "__init__.py",
        "dense_image_warp.py",
        "distance_transform.py",
        "distort_image_ops.py",
        "filters.py",
        "transform_ops.py",
        "translate_ops.py",
        "utils.py",
        "sparse_image_warp.py",
        "interpolate_spline.py",
        "connected_components.py",
        "resampler_ops.py",
    ]),
    data = [
        ":sparse_image_warp_test_data",
        "//tensorflow_addons/custom_ops/image:_distort_image_ops.so",
        "//tensorflow_addons/custom_ops/image:_image_ops.so",
        "//tensorflow_addons/custom_ops/image:_resampler_ops.so",
        "//tensorflow_addons/utils",
    ],
)

filegroup(
    name = "sparse_image_warp_test_data",
    srcs = glob(["test_data/*.png"]),
)

