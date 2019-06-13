licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "callbacks",
    srcs = [
        "__init__.py",
        "OneCycleLR.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_addons/utils",
    ],
)

py_test(
    name = "OneCycleLR_test",
    size = "medium",
    srcs = [
        "OneCycleLR_test.py",
    ],
    main = "OneCycleLR_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":callbacks",
    ],
)
