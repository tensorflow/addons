licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "activations",
    srcs = [
        "__init__.py",
        "gelu.py",
        "hardshrink.py",
        "lisht.py",
        "mish.py",
        "rrelu.py",
        "softshrink.py",
        "sparsemax.py",
        "tanhshrink.py",
    ],
    data = [
        "//tensorflow_addons:options.py",
        "//tensorflow_addons/custom_ops/activations:_activation_ops.so",
        "//tensorflow_addons/utils",
    ],
)

py_test(
    name = "activations_test",
    size = "small",
    srcs = glob(["*_test.py"]),
    main = "run_all_test.py",
    deps = [
        ":activations",
    ],
)
