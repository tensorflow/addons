licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "callbacks",
    srcs = [
        "__init__.py",
        "map_callback.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_addons/utils",
    ],
)

py_test(
    name = "map_callback_test",
    size = "small",
    srcs = [
        "map_callback_test.py",
    ],
    main = "map_callback_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":callbacks",
    ],
)
