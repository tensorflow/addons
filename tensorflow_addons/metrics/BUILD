licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "metrics",
    srcs = [
        "__init__.py",
        "cohens_kappa.py",
        "f_scores.py",
        "multilabel_confusion_matrix.py",
        "r_square.py",
        "utils.py",
        "crf_accuracy.py"
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

py_test(
    name = "r_square_test",
    size = "small",
    srcs = [
        "r_square_test.py",
    ],
    main = "r_square_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":metrics",
    ],
)

py_test(
    name = "f1_test",
    size = "small",
    srcs = [
        "f1_test.py",
    ],
    main = "f1_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":metrics",
    ],
)

py_test(
    name = "fbeta_test",
    size = "small",
    srcs = [
        "fbeta_test.py",
    ],
    main = "fbeta_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":metrics",
    ],
)

py_test(
    name = "multilabel_confusion_matrix_test",
    size = "small",
    srcs = [
        "multilabel_confusion_matrix_test.py",
    ],
    main = "multilabel_confusion_matrix_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":metrics",
    ],
)
