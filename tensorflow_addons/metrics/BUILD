licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "metrics",
    srcs = [
        "__init__.py",
        "cohens_kappa.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_addons/utils",
    ],
)

py_test(
    name = "cohens_kappa_test",
    size = "small",
    srcs = [
        "cohens_kappa_test.py",
    ],
    main = "cohens_kappa_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":metrics",
    ],
)

py_library(
    name = "metrics",
    srcs = [
        "__init__.py",
        "f1_scores.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_addons/utils",
    ],
)

py_test(
    name = "f1_micro_test",
    size = "small",
    srcs = [
        "f1_micro_test.py",
    ],
    main = "f1_micro_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":metrics",
    ],
)

py_test(
    name = "f1_macro_and_weighted_test",
    size = "small",
    srcs = [
        "f1_macro_and_weighted_test.py",
    ],
    main = "f1_macro_and_weighted_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":metrics",
    ],
)
