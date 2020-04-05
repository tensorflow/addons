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
    name = "serialization_test",
    size = "small",
    srcs = glob(["*_test.py"]),
    main = "run_all_test.py",
    deps = [
        ":testing",
    ],
)
