licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "layers_py",
    srcs = [
        "__init__.py",
        "maxout.py",
        "normalizations.py",
        "poincare.py",
        "sparsemax.py",
        "wrappers.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_addons/utils:utils_py",
        "//tensorflow_addons/activations:activations_py",
    ],
)

py_test(
    name = "layers_wrappers_py_test",
    size = "small",
    srcs = [
        "wrappers_test.py",
    ],
    main = "wrappers_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":layers_py",
    ]
)

py_test(
    name = "sparsemax_py_test",
    size = "small",
    srcs = [
        "sparsemax_test.py",
    ],
    main = "sparsemax_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":layers_py",
    ],
)

py_test(
    name = "maxout_py_test",
    size = "small",
    srcs = [
        "maxout_test.py",
    ],
    main = "maxout_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":layers_py",
    ]
)

py_test(
    name = "layers_normalizations_py_test",
    size= "small",
    srcs = [
        "normalizations_test.py",
    ],
    main = "normalizations_test.py",
    srcs_version = "PY2AND3",
    deps = [
            ":layers_py",
        ]
)
