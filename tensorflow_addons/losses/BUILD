licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "losses",
    srcs = [
        "__init__.py",
        "contrastive.py",
        "focal_loss.py",
        "giou_loss.py",
        "kappa_loss.py",
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

py_test(
    name = "losses_test",
    size = "small",
    srcs = glob(["*_test.py"]),
    main = "run_all_test.py",
    deps = [
        ":losses",
    ],
)

py_test(
    name = "kappa_loss_test",
    size = "small",
    srcs = [
        "kappa_loss_test.py",
    ],
    main = "kappa_loss_test.py",
    deps = [
        ":losses",
    ],
)
