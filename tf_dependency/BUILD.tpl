package(default_visibility = ["//visibility:public"])

cc_library(
    name = "tf_header_lib",
    hdrs = [":tf_header_include"],
    includes = ["include"],
    visibility = ["//visibility:public"],
)

# FIXME: Read shared library name from the installed python package.
# See https://github.com/tensorflow/addons/pull/130 for why we're linking
# to framework.so.1  in the meantime.

cc_library(
    name = "libtensorflow_framework",
    srcs = [":libtensorflow_framework.so.1"],
    #data = ["lib/libtensorflow_framework.so.1"],
    visibility = ["//visibility:public"],
)

%{TF_HEADER_GENRULE}
%{TF_SHARED_LIBRARY_GENRULE}