licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "sparse",
    srcs = glob(["*.py"]),
    data = [
        "//tensorflow_addons/custom_ops/sparse:_sparse_ops.so",
        "//tensorflow_addons/utils",
    ],
)

py_test(
    name = "sparse_test",
    srcs = glob(["tests/*"]),
    main = "tests/run_all_test.py",
    deps = [
        ":sparse",
    ],
)
