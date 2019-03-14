licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "losses_py",
    srcs = [
        "__init__.py",
        "lifted.py",
        "metric_learning.py",
        "sparsemax_loss.py",
        "triplet.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_addons/utils:utils_py",
        "//tensorflow_addons/activations:activations_py",
    ],
)

py_test(
    name = "sparsemax_loss_py_test",
    size = "small",
    srcs = [
        "sparsemax_loss_test.py",
    ],
    main = "sparsemax_loss_test.py",
    deps = [
        ":losses_py",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "triplet_py_test",
    size = "small",
    srcs = [
        "triplet_test.py",
    ],
    main = "triplet_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":losses_py",
    ],
)

py_test(
    name = "lifted_py_test",
    size = "small",
    srcs = [
        "lifted_test.py",
    ],
    main = "lifted_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":losses_py",
    ],
)
