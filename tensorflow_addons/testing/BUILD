licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "testing",
    srcs = glob(["*.py"]),
)

py_test(
    name = "serialization_test",
    size = "small",
    srcs = glob(["tests/*"]),
    main = "tests/run_all_test.py",
    deps = [
        ":testing",
    ],
)
