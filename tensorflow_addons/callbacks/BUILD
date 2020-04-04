licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "callbacks",
    srcs = [
        "__init__.py",
        "average_model_checkpoint.py",
        "time_stopping.py",
        "tqdm_progress_bar.py",
    ],
    deps = [
        "//tensorflow_addons/optimizers",
        "//tensorflow_addons/utils",
    ],
)

py_test(
    name = "callbacks_test",
    size = "small",
    srcs = glob(["*_test.py"]),
    main = "run_all_test.py",
    deps = [
        ":callbacks",
    ],
)
