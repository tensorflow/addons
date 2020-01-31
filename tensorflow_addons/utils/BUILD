licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "utils",
    srcs = ([
        "__init__.py",
        "ensure_tf_install",
        "keras_utils.py",
        "test_utils.py",
        "resource_loader.py",
        "types.py",
    ]),
)

py_test(
    name = "keras_utils_test",
    size = "small",
    srcs = [
        "keras_utils_test.py",
    ],
    main = "keras_utils_test.py",
    deps = [
        ":utils",
    ],
)
