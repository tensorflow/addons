licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "utils",
    srcs = ([
        "__init__.py",
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

py_test(
    name = "test_utils_test",
    size = "medium",
    srcs = [
        "test_utils_test.py",
    ],
    main = "test_utils_test.py",
    deps = [
        ":utils",
    ],
)
