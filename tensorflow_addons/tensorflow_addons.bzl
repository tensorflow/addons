load("@local_config_tf//:build_defs.bzl", "CPLUSPLUS_VERSION", "D_GLIBCXX_USE_CXX11_ABI")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda", "if_cuda_is_configured")

def custom_op_library(
        name,
        srcs = [],
        cuda_srcs = [],
        deps = [],
        cuda_deps = [],
        copts = [],
        **kwargs):
    deps = deps + [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ]

    if cuda_srcs:
        copts = copts + if_cuda(["-DGOOGLE_CUDA=1"])
        cuda_copts = copts + if_cuda_is_configured([
            "-x cuda",
            "-nvcc_options=relaxed-constexpr",
            "-nvcc_options=ftz=true",
        ])
        cuda_deps = deps + if_cuda_is_configured(cuda_deps) + if_cuda_is_configured([
            "@local_config_cuda//cuda:cuda_headers",
            "@local_config_cuda//cuda:cudart_static",
        ])
        basename = name.split(".")[0]
        native.cc_library(
            name = basename + "_gpu",
            srcs = cuda_srcs,
            deps = cuda_deps,
            copts = cuda_copts,
            alwayslink = 1,
            **kwargs
        )
        deps = deps + if_cuda_is_configured([":" + basename + "_gpu"])

    copts = copts + select({
        "//tensorflow_addons:windows": [
            "/DEIGEN_STRONG_INLINE=inline",
            "-DTENSORFLOW_MONOLITHIC_BUILD",
            "/D_USE_MATH_DEFINES",
            "/DPLATFORM_WINDOWS",
            "/DEIGEN_HAS_C99_MATH",
            "/DTENSORFLOW_USE_EIGEN_THREADPOOL",
            "/DEIGEN_AVOID_STL_ARRAY",
            "/Iexternal/gemmlowp",
            "/wd4018",
            "/wd4577",
            "/DNOGDI",
            "/UTF_COMPILE_LIBRARY",
        ],
        "//conditions:default": ["-pthread", CPLUSPLUS_VERSION, D_GLIBCXX_USE_CXX11_ABI],
    })

    native.cc_binary(
        name = name,
        srcs = srcs,
        copts = copts,
        linkshared = 1,
        features = select({
            "//tensorflow_addons:windows": ["windows_export_all_symbols"],
            "//conditions:default": [],
        }),
        deps = deps,
        **kwargs
    )
