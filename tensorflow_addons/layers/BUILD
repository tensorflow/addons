licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_test(
    name = "layers_poincare_py_test",
    srcs = [
        "python/layers/poincare_normalize_test.py",
    ],
    main = "python/layers/poincare_normalize_test.py",
    srcs_version = "PY2AND3",
)

py_test(
    name = "layers_wrappers_py_test",
    srcs = [
        "python/layers/wrappers_test.py",
    ],
    main = "python/layers/wrappers_test.py",
    srcs_version = "PY2AND3",
)

py_library(
    name = "layers_py",
    srcs = ([
        "__init__.py",
        "python/__init__.py",
        "python/layers/__init__.py",
        "python/layers/poincare_normalize.py",
        "python/layers/wrappers.py",
    ]),
    srcs_version = "PY2AND3",
)