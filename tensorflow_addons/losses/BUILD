licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "losses",
    srcs = [
        "__init__.py",
        "contrastive.py",
        "crf_loss.py",
        "focal_loss.py",
        "giou_loss.py",
        "lifted.py",
        "metric_learning.py",
        "npairs.py",
        "sparsemax_loss.py",
        "triplet.py",
    ],
    deps = [
        "//tensorflow_addons/activations",
        "//tensorflow_addons/layers",
        "//tensorflow_addons/utils",
    ],
)

py_test(
    name = "contrastive_test",
    size = "small",
    srcs = [
        "contrastive_test.py",
    ],
    main = "contrastive_test.py",
    deps = [
        ":losses",
    ],
)

py_test(
    name = "focal_loss_test",
    size = "small",
    srcs = [
        "focal_loss_test.py",
    ],
    main = "focal_loss_test.py",
    deps = [
        ":losses",
    ],
)

py_test(
    name = "giou_loss_test",
    size = "small",
    srcs = [
        "giou_loss_test.py",
    ],
    main = "giou_loss_test.py",
    deps = [
        ":losses",
    ],
)

py_test(
    name = "npairs_test",
    size = "small",
    srcs = [
        "npairs_test.py",
    ],
    main = "npairs_test.py",
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
)

py_test(
    name = "triplet_test",
    size = "small",
    srcs = [
        "triplet_test.py",
    ],
    main = "triplet_test.py",
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
    deps = [
        ":losses",
    ],
)

py_test(
    name = "metric_test",
    size = "small",
    srcs = [
        "metric_test.py",
    ],
    main = "metric_test.py",
    deps = [
        ":losses",
    ],
)

py_test(
    name = "crf_loss_test",
    size = "small",
    srcs = [
        "crf_loss_test.py",
    ],
    main = "crf_loss_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":losses",
        "//tensorflow_addons/layers",
    ],
)
