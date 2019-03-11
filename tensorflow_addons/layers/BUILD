licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "layers_py",
    srcs = [
        "__init__.py",
        "python/__init__.py",
        "python/maxout.py",
        "python/normalizations.py",
        "python/poincare.py",
        "python/sparsemax.py",
        "python/wrappers.py",
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
        "python/wrappers_test.py",
    ],
    main = "python/wrappers_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":layers_py",
    ]
)

py_test(
    name = "sparsemax_py_test",
    size = "small",
    srcs = [
        "python/sparsemax_test.py",
    ],
    main = "python/sparsemax_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":layers_py",
    ],
)

py_test(
    name = "maxout_py_test",
    size = "small",
    srcs = [
        "python/maxout_test.py",
    ],
    main = "python/maxout_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":layers_py",
    ]
)

py_test(
    name = "layers_normalizations_py_test",
    size= "small",
    srcs = [
        "python/normalizations_test.py",
    ],
    main = "python/normalizations_test.py",
    srcs_version = "PY2AND3",
    deps = [
            ":layers_py",
        ]
)
