licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "utils",
    srcs = ([
        "__init__.py",
        "keras_utils.py",
        "test_utils.py",
        "resource_loader.py",
    ]),
    srcs_version = "PY2AND3",
)

py_test(
    name = "keras_utils_test",
    size = "small",
    srcs = [
        "keras_utils_test.py",
    ],
    main = "keras_utils_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":utils",
    ],
)
