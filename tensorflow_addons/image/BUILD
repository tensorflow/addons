licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "image",
    srcs = glob(["*.py"]),
    data = [
        ":sparse_image_warp_test_data",
        "//tensorflow_addons/custom_ops/image:_distort_image_ops.so",
        "//tensorflow_addons/custom_ops/image:_image_ops.so",
        "//tensorflow_addons/custom_ops/image:_resampler_ops.so",
        "//tensorflow_addons/testing",
        "//tensorflow_addons/utils",
    ],
)

filegroup(
    name = "sparse_image_warp_test_data",
    srcs = glob(["tests/test_data/*.png"]),
)

py_test(
    name = "image_test",
    size = "small",
    srcs = glob(["tests/*"]),
    main = "tests/run_all_test.py",
    deps = [
        ":image",
    ],
)
