licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "losses",
    srcs = [
        "__init__.py",
        "focal_loss.py",
        "lifted.py",
        "metric_learning.py",
        "sparsemax_loss.py",
        "triplet.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_addons/utils:utils",
        "//tensorflow_addons/activations:activations",
    ],
)

py_test(
    name = "focal_loss_test",
    size = "small",
    srcs = [
        "focal_loss_test.py",
    ],
    main = "focal_loss_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":losses",
    ],
)

py_test(
    name = "sparsemax_loss_test",
    size = "small",
    srcs = [
        "sparsemax_loss_test.py",
    ],
    main = "sparsemax_loss_test.py",
    deps = [
        ":losses",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "triplet_test",
    size = "small",
    srcs = [
        "triplet_test.py",
    ],
    main = "triplet_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":losses",
    ],
)

py_test(
    name = "lifted_test",
    size = "small",
    srcs = [
        "lifted_test.py",
    ],
    main = "lifted_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":losses",
    ],
)
