licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "training",
    srcs = [
        "__init__.py",
        "gradient_checkpointing.py",
    ],
    data = [
        "//tensorflow_addons/utils",
    ],
)

py_test(
    name = "training_test",
    size = "small",
    srcs = glob(["*_test.py"]),
    main = "run_all_test.py",
    deps = [
        ":training",
    ],
)
