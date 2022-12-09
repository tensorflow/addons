load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//build_deps/tf_dependency:tf_configure.bzl", "tf_configure")
load("//build_deps/toolchains/gpu:cuda_configure.bzl", "cuda_configure")

http_archive(
    name = "cub_archive",
    build_file = "//build_deps/toolchains/gpu:cub.BUILD",
    sha256 = "6bfa06ab52a650ae7ee6963143a0bbc667d6504822cbd9670369b598f18c58c3",
    strip_prefix = "cub-1.8.0",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/NVlabs/cub/archive/1.8.0.zip",
        "https://github.com/NVlabs/cub/archive/1.8.0.zip",
    ],
)

tf_configure(
    name = "local_config_tf",
)

cuda_configure(name = "local_config_cuda")

http_archive(
    name = "org_tensorflow",
    sha256 = "99c732b92b1b37fc243a559e02f9aef5671771e272758aa4aec7f34dc92dac48",
    strip_prefix = "tensorflow-2.11.0",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.11.0.tar.gz",
    ],
)

load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")

tf_workspace3()

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")

tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")

tf_workspace1()

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")

tf_workspace0()
