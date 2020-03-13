licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "testing",
    srcs = [
        "__init__.py",
        "serialization.py",
    ],
)

py_test(
    name = "test_serialization",
    size = "small",
    srcs = [
        "test_serialization.py",
    ],
    main = "test_serialization.py",
    deps = [
        ":testing",
    ],
)
