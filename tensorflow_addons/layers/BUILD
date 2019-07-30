licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "layers",
    srcs = [
        "__init__.py",
        "maxout.py",
        "normalizations.py",
        "poincare.py",
        "sparsemax.py",
        "wrappers.py",
        "crf.py"
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_addons/activations",
        "//tensorflow_addons/utils",
        "//tensorflow_addons/text",
    ],
)

py_test(
    name = "layers_wrappers_test",
    size = "small",
    srcs = [
        "wrappers_test.py",
    ],
    main = "wrappers_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":layers",
    ],
)

py_test(
    name = "sparsemax_test",
    size = "small",
    srcs = [
        "sparsemax_test.py",
    ],
    main = "sparsemax_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":layers",
    ],
)

py_test(
    name = "maxout_test",
    size = "small",
    srcs = [
        "maxout_test.py",
    ],
    main = "maxout_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":layers",
    ],
)

py_test(
    name = "layers_normalizations_test",
    size = "small",
    srcs = [
        "normalizations_test.py",
    ],
    main = "normalizations_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":layers",
    ],
)
