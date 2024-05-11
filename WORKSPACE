load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//build_deps/tf_dependency:tf_configure.bzl", "tf_configure")
load("//build_deps/toolchains/gpu:cuda_configure.bzl", "cuda_configure")

tf_configure(
    name = "local_config_tf",
)

http_archive(
    name = "org_tensorflow",
    patch_cmds = [
        """sed -i.bak 's/cython-3.0.3/cython-3.0.0a11/g' tensorflow/workspace2.bzl""",
        """sed -i.bak 's/3.0.3.tar.gz/3.0.0a11.tar.gz/g' tensorflow/workspace2.bzl""",
        """sed -i.bak 's/0c2eae8a4ceab7955be1e11a4ddc5dcc3aa06ce22ad594262f1555b9d10667f0/08dbdb6aa003f03e65879de8f899f87c8c718cd874a31ae9c29f8726da2f5ab0/g' tensorflow/workspace2.bzl""",
    ],
    sha256 = "c729e56efc945c6df08efe5c9f5b8b89329c7c91b8f40ad2bb3e13900bd4876d",
    strip_prefix = "tensorflow-2.16.1",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.16.1.tar.gz",
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
