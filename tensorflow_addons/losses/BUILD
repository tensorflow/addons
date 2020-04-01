licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "losses",
    srcs = [
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
    deps = [
        "//tensorflow_addons/activations",
        "//tensorflow_addons/utils",
    ],
)
