load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//build_deps/tf_dependency:tf_configure.bzl", "tf_configure")
load("//build_deps/toolchains/gpu:cuda_configure.bzl", "cuda_configure")

tf_configure(
    name = "local_config_tf",
)

http_archive(
    name = "org_tensorflow",
    patches = [
        "//build_deps/tf_dependency:tf.patch",
    ],
    sha256 = "9cec5acb0ecf2d47b16891f8bc5bc6fbfdffe1700bdadc0d9ebe27ea34f0c220",
    strip_prefix = "tensorflow-2.15.0",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.15.0.tar.gz",
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
