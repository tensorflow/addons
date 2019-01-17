licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "layers_py",
    srcs = [
        "__init__.py",
        "python/__init__.py",
        "python/maxout.py",
        "python/wrappers.py",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "layers_wrappers_py_test",
    srcs = [
        "python/wrappers_test.py",
    ],
    main = "python/wrappers_test.py",
    deps = [
        ":layers_py",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "maxout_py_test",
    size = "small",
    srcs = [
        "python/maxout_test.py",
    ],
    main = "python/maxout_test.py",
    deps = [
        ":layers_py",
    ],
    srcs_version = "PY2AND3",
)

