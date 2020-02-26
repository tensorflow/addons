licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

# TODO: Build npairs for windows after SparseMatMul issue is solved
# https://github.com/tensorflow/addons/issues/838
py_library(
    name = "losses",
    srcs = select({
        "//tensorflow_addons:windows": [
            "__init__.py",
            "contrastive.py",
            "focal_loss.py",
            "giou_loss.py",
            "lifted.py",
            "metric_learning.py",
            "quantiles.py",
            "sparsemax_loss.py",
            "triplet.py",
        ],
        "//conditions:default": [
            "__init__.py",
            "conter_loss.py",
            "contrastive.py",
            "focal_loss.py",
            "giou_loss.py",
            "lifted.py",
            "metric_learning.py",
            "npairs.py",
            "quantiles.py",
            "sparsemax_loss.py",
            "triplet.py",
        ],
    }),
    deps = [
        "//tensorflow_addons/activations",
        "//tensorflow_addons/utils",
    ],
)

py_test(
    name = "conter_loss_test",
    size = "small",
    srcs = [
        "conter_loss_test.py",
    ],
    main = "conter_loss_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":losses",
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

# Temporarily disable on windows
py_test(
    name = "npairs_test",
    size = "small",
    srcs = select({
        "//tensorflow_addons:windows": ["npairs_dummy_test.py"],
        "//conditions:default": ["npairs_test.py"],
    }),
    main = select({
        "//tensorflow_addons:windows": "npairs_dummy_test.py",
        "//conditions:default": "npairs_test.py",
    }),
    deps = [
        ":losses",
    ],
)

py_test(
    name = "quantiles_test",
    size = "small",
    srcs = [
        "quantiles_test.py",
    ],
    main = "quantiles_test.py",
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
