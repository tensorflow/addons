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
