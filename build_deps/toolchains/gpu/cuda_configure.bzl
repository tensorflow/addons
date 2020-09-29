# -*- Python -*-
"""Repository rule for CUDA autoconfiguration.
`cuda_configure` depends on the following environment variables:
  * `TF_NEED_CUDA`: Whether to enable building with CUDA.
  * `GCC_HOST_COMPILER_PATH`: The GCC host compiler path
  * `TF_CUDA_CLANG`: Whether to use clang as a cuda compiler.
  * `CLANG_CUDA_COMPILER_PATH`: The clang compiler path that will be used for
    both host and device code compilation if TF_CUDA_CLANG is 1.
  * `TF_CUDA_PATHS`: The base paths to look for CUDA and cuDNN. Default is
    `/usr/local/cuda,usr/`.
  * `CUDA_TOOLKIT_PATH`: The path to the CUDA toolkit. Default is
    `/usr/local/cuda`.
  * `TF_CUDA_VERSION`: The version of the CUDA toolkit. If this is blank, then
    use the system default.
  * `TF_CUDNN_VERSION`: The version of the cuDNN library.
  * `CUDNN_INSTALL_PATH`: The path to the cuDNN library. Default is
    `/usr/local/cuda`.
  * `TF_CUDA_COMPUTE_CAPABILITIES`: The CUDA compute capabilities. Default is
    `3.5,5.2`.
  * `PYTHON_BIN_PATH`: The python binary path
"""

load(
    "@bazel_tools//tools/cpp:lib_cc_configure.bzl",
    "escape_string",
    "get_env_var",
)
load(
    "@bazel_tools//tools/cpp:windows_cc_configure.bzl",
    "find_msvc_tool",
    "find_vc_path",
    "setup_vc_env_vars",
)

_GCC_HOST_COMPILER_PATH = "GCC_HOST_COMPILER_PATH"

_CLANG_CUDA_COMPILER_PATH = "CLANG_CUDA_COMPILER_PATH"

_CUDA_TOOLKIT_PATH = "CUDA_TOOLKIT_PATH"

_TF_CUDA_VERSION = "TF_CUDA_VERSION"

_TF_CUDNN_VERSION = "TF_CUDNN_VERSION"

_CUDNN_INSTALL_PATH = "CUDNN_INSTALL_PATH"

_TF_CUDA_COMPUTE_CAPABILITIES = "TF_CUDA_COMPUTE_CAPABILITIES"

_TF_DOWNLOAD_CLANG = "TF_DOWNLOAD_CLANG"

_PYTHON_BIN_PATH = "PYTHON_BIN_PATH"

_DEFAULT_CUDA_COMPUTE_CAPABILITIES = [
    "3.5",
    "5.2",
]

def _get_python_bin(repository_ctx):
    """Gets the python bin path."""
    python_bin = repository_ctx.os.environ.get(_PYTHON_BIN_PATH)
    if python_bin != None:
        return python_bin
    python_bin_name = "python.exe" if _is_windows(repository_ctx) else "python"
    python_bin_path = repository_ctx.which(python_bin_name)
    if python_bin_path != None:
        return str(python_bin_path)
    auto_configure_fail(
        "Cannot find python in PATH, please make sure " +
        "python is installed and add its directory in PATH, or --define " +
        "%s='/something/else'.\nPATH=%s" % (
            _PYTHON_BIN_PATH,
            repository_ctx.os.environ.get("PATH", ""),
        ),
    )

def _get_nvcc_tmp_dir_for_windows(repository_ctx):
    """Return the tmp directory for nvcc to generate intermediate source files."""
    escaped_tmp_dir = escape_string(
        get_env_var(repository_ctx, "TMP", "C:\\Windows\\Temp").replace(
            "\\",
            "\\\\",
        ),
    )
    return escaped_tmp_dir + "\\\\nvcc_inter_files_tmp_dir"

def _get_msvc_compiler(repository_ctx):
    vc_path = find_vc_path(repository_ctx)
    return find_msvc_tool(repository_ctx, vc_path, "cl.exe").replace("\\", "/")

def _get_win_cuda_defines(repository_ctx):
    """Return CROSSTOOL defines for Windows"""

    # If we are not on Windows, return empty vaules for Windows specific fields.
    # This ensures the CROSSTOOL file parser is happy.
    if not _is_windows(repository_ctx):
        return dict({
            "%{msvc_env_tmp}": "",
            "%{msvc_env_path}": "",
            "%{msvc_env_include}": "",
            "%{msvc_env_lib}": "",
            "%{msvc_cl_path}": "",
            "%{msvc_ml_path}": "",
            "%{msvc_link_path}": "",
            "%{msvc_lib_path}": "",
            "%{cxx_builtin_include_directory}": "",
        })

    vc_path = find_vc_path(repository_ctx)
    if not vc_path:
        auto_configure_fail(
            "Visual C++ build tools not found on your machine. " +
            "Please check your installation following https://docs.bazel.build/versions/master/windows.html#using",
        )
        return {}

    env = setup_vc_env_vars(repository_ctx, vc_path)
    escaped_paths = escape_string(env["PATH"])
    escaped_include_paths = escape_string(env["INCLUDE"])
    escaped_lib_paths = escape_string(env["LIB"])
    escaped_tmp_dir = escape_string(
        get_env_var(repository_ctx, "TMP", "C:\\Windows\\Temp").replace(
            "\\",
            "\\\\",
        ),
    )

    msvc_cl_path = _get_python_bin(repository_ctx)
    msvc_ml_path = find_msvc_tool(repository_ctx, vc_path, "ml64.exe").replace(
        "\\",
        "/",
    )
    msvc_link_path = find_msvc_tool(repository_ctx, vc_path, "link.exe").replace(
        "\\",
        "/",
    )
    msvc_lib_path = find_msvc_tool(repository_ctx, vc_path, "lib.exe").replace(
        "\\",
        "/",
    )

    # nvcc will generate some temporary source files under %{nvcc_tmp_dir}
    # The generated files are guranteed to have unique name, so they can share the same tmp directory
    escaped_cxx_include_directories = [
        "cxx_builtin_include_directory: \"%s\"" %
        _get_nvcc_tmp_dir_for_windows(repository_ctx),
    ]
    for path in escaped_include_paths.split(";"):
        if path:
            escaped_cxx_include_directories.append(
                "cxx_builtin_include_directory: \"%s\"" % path,
            )

    return {
        "%{msvc_env_tmp}": escaped_tmp_dir,
        "%{msvc_env_path}": escaped_paths,
        "%{msvc_env_include}": escaped_include_paths,
        "%{msvc_env_lib}": escaped_lib_paths,
        "%{msvc_cl_path}": msvc_cl_path,
        "%{msvc_ml_path}": msvc_ml_path,
        "%{msvc_link_path}": msvc_link_path,
        "%{msvc_lib_path}": msvc_lib_path,
        "%{cxx_builtin_include_directory}": "\n".join(escaped_cxx_include_directories),
    }

def find_cc(repository_ctx):
    """Find the C++ compiler."""
    if _is_windows(repository_ctx):
        return _get_msvc_compiler(repository_ctx)

    target_cc_name = "gcc"
    cc_path_envvar = _GCC_HOST_COMPILER_PATH
    cc_name = target_cc_name

    if cc_path_envvar in repository_ctx.os.environ:
        cc_name_from_env = repository_ctx.os.environ[cc_path_envvar].strip()
        if cc_name_from_env:
            cc_name = cc_name_from_env
    if cc_name.startswith("/"):
        # Absolute path, maybe we should make this supported by our which function.
        return cc_name
    cc = repository_ctx.which(cc_name)
    if cc == None:
        fail(("Cannot find {}, either correct your path or set the {}" +
              " environment variable").format(target_cc_name, cc_path_envvar))
    return cc

_INC_DIR_MARKER_BEGIN = "#include <...>"

# OSX add " (framework directory)" at the end of line, strip it.
_OSX_FRAMEWORK_SUFFIX = " (framework directory)"

_OSX_FRAMEWORK_SUFFIX_LEN = len(_OSX_FRAMEWORK_SUFFIX)

def _cxx_inc_convert(path):
    """Convert path returned by cc -E xc++ in a complete path."""
    path = path.strip()
    if path.endswith(_OSX_FRAMEWORK_SUFFIX):
        path = path[:-_OSX_FRAMEWORK_SUFFIX_LEN].strip()
    return path

def _normalize_include_path(repository_ctx, path):
    """Normalizes include paths before writing them to the crosstool.
      If path points inside the 'crosstool' folder of the repository, a relative
      path is returned.
      If path points outside the 'crosstool' folder, an absolute path is returned.
      """
    path = str(repository_ctx.path(path))
    crosstool_folder = str(repository_ctx.path(".").get_child("crosstool"))

    if path.startswith(crosstool_folder):
        # We drop the path to "$REPO/crosstool" and a trailing path separator.
        return path[len(crosstool_folder) + 1:]
    return path

def _get_cxx_inc_directories_impl(repository_ctx, cc, lang_is_cpp):
    """Compute the list of default C or C++ include directories."""
    if lang_is_cpp:
        lang = "c++"
    else:
        lang = "c"
    result = repository_ctx.execute([cc, "-E", "-x" + lang, "-", "-v"])
    index1 = result.stderr.find(_INC_DIR_MARKER_BEGIN)
    if index1 == -1:
        return []
    index1 = result.stderr.find("\n", index1)
    if index1 == -1:
        return []
    index2 = result.stderr.rfind("\n ")
    if index2 == -1 or index2 < index1:
        return []
    index2 = result.stderr.find("\n", index2 + 1)
    if index2 == -1:
        inc_dirs = result.stderr[index1 + 1:]
    else:
        inc_dirs = result.stderr[index1 + 1:index2].strip()

    return [
        _normalize_include_path(repository_ctx, _cxx_inc_convert(p))
        for p in inc_dirs.split("\n")
    ]

def get_cxx_inc_directories(repository_ctx, cc):
    """Compute the list of default C and C++ include directories."""

    # For some reason `clang -xc` sometimes returns include paths that are
    # different from the ones from `clang -xc++`. (Symlink and a dir)
    # So we run the compiler with both `-xc` and `-xc++` and merge resulting lists
    includes_cpp = _get_cxx_inc_directories_impl(repository_ctx, cc, True)
    includes_c = _get_cxx_inc_directories_impl(repository_ctx, cc, False)

    return includes_cpp + [
        inc
        for inc in includes_c
        if inc not in includes_cpp
    ]

def auto_configure_fail(msg):
    """Output failure message when cuda configuration fails."""
    red = "\033[0;31m"
    no_color = "\033[0m"
    fail("\n%sCuda Configuration Error:%s %s\n" % (red, no_color, msg))

# END cc_configure common functions (see TODO above).

def _host_compiler_includes(repository_ctx, cc):
    """Generates the cxx_builtin_include_directory entries for gcc inc dirs.
      Args:
        repository_ctx: The repository context.
        cc: The path to the gcc host compiler.
      Returns:
        A string containing the cxx_builtin_include_directory for each of the gcc
        host compiler include directories, which can be added to the CROSSTOOL
        file.
      """
    inc_dirs = get_cxx_inc_directories(repository_ctx, cc)
    inc_entries = []
    for inc_dir in inc_dirs:
        inc_entries.append("  cxx_builtin_include_directory: \"%s\"" % inc_dir)
    return "\n".join(inc_entries)

def _cuda_include_path(repository_ctx, cuda_config):
    """Generates the cxx_builtin_include_directory entries for cuda inc dirs.
      Args:
        repository_ctx: The repository context.
        cc: The path to the gcc host compiler.
      Returns:
        A string containing the cxx_builtin_include_directory for each of the gcc
        host compiler include directories, which can be added to the CROSSTOOL
        file.
      """
    nvcc_path = repository_ctx.path("%s/bin/nvcc%s" % (
        cuda_config.cuda_toolkit_path,
        ".exe" if cuda_config.cpu_value == "Windows" else "",
    ))
    result = repository_ctx.execute([
        nvcc_path,
        "-v",
        "/dev/null",
        "-o",
        "/dev/null",
    ])
    target_dir = ""
    for one_line in result.stderr.splitlines():
        if one_line.startswith("#$ _TARGET_DIR_="):
            target_dir = (
                cuda_config.cuda_toolkit_path + "/" + one_line.replace(
                    "#$ _TARGET_DIR_=",
                    "",
                ) + "/include"
            )
    inc_entries = []
    if target_dir != "":
        inc_entries.append("  cxx_builtin_include_directory: \"%s\"" % target_dir)
    default_include = cuda_config.cuda_toolkit_path + "/include"
    inc_entries.append(
        "  cxx_builtin_include_directory: \"%s\"" % default_include,
    )
    return "\n".join(inc_entries)

def enable_cuda(repository_ctx):
    if "TF_NEED_CUDA" in repository_ctx.os.environ:
        enable_cuda = repository_ctx.os.environ["TF_NEED_CUDA"].strip()
        return enable_cuda == "1"
    return False

def matches_version(environ_version, detected_version):
    """Checks whether the user-specified version matches the detected version.
      This function performs a weak matching so that if the user specifies only
      the
      major or major and minor versions, the versions are still considered
      matching
      if the version parts match. To illustrate:
          environ_version  detected_version  result
          -----------------------------------------
          5.1.3            5.1.3             True
          5.1              5.1.3             True
          5                5.1               True
          5.1.3            5.1               False
          5.2.3            5.1.3             False
      Args:
        environ_version: The version specified by the user via environment
          variables.
        detected_version: The version autodetected from the CUDA installation on
          the system.
      Returns: True if user-specified version matches detected version and False
        otherwise.
    """
    environ_version_parts = environ_version.split(".")
    detected_version_parts = detected_version.split(".")
    if len(detected_version_parts) < len(environ_version_parts):
        return False
    for i, part in enumerate(detected_version_parts):
        if i >= len(environ_version_parts):
            break
        if part != environ_version_parts[i]:
            return False
    return True

def find_cuda_define(repository_ctx, header_dir, header_file, define):
    """Returns the value of a #define in a header file.
      Greps through a header file and returns the value of the specified #define.
      If the #define is not found, then raise an error.
      Args:
        repository_ctx: The repository context.
        header_dir: The directory containing the header file.
        header_file: The header file name.
        define: The #define to search for.
      Returns:
        The value of the #define found in the header.
      """

    # Confirm location of the header and grep for the line defining the macro.
    h_path = repository_ctx.path("%s/%s" % (header_dir, header_file))
    if not h_path.exists:
        auto_configure_fail("Cannot find %s at %s" % (header_file, str(h_path)))
    result = repository_ctx.execute(
        # Grep one more lines as some #defines are split into two lines.
        [
            "grep",
            "--color=never",
            "-A1",
            "-E",
            define,
            str(h_path),
        ],
    )
    if result.stderr:
        auto_configure_fail("Error reading %s: %s" % (str(h_path), result.stderr))

    # Parse the version from the line defining the macro.
    if result.stdout.find(define) == -1:
        auto_configure_fail(
            "Cannot find line containing '%s' in %s" % (define, h_path),
        )

    # Split results to lines
    lines = result.stdout.split("\n")
    num_lines = len(lines)
    for l in range(num_lines):
        line = lines[l]
        if define in line:  # Find the line with define
            version = line
            if l != num_lines - 1 and line[-1] == "\\":  # Add next line, if multiline
                version = version[:-1] + lines[l + 1]
            break

    # Remove any comments
    version = version.split("//")[0]

    # Remove define name
    version = version.replace(define, "").strip()

    # Remove the code after the version number.
    version_end = version.find(" ")
    if version_end != -1:
        if version_end == 0:
            auto_configure_fail(
                "Cannot extract the version from line containing '%s' in %s" %
                (define, str(h_path)),
            )
        version = version[:version_end].strip()
    return version

def compute_capabilities(repository_ctx):
    """Returns a list of strings representing cuda compute capabilities."""
    if _TF_CUDA_COMPUTE_CAPABILITIES not in repository_ctx.os.environ:
        return _DEFAULT_CUDA_COMPUTE_CAPABILITIES
    capabilities_str = repository_ctx.os.environ[_TF_CUDA_COMPUTE_CAPABILITIES]
    capabilities = capabilities_str.split(",")
    for capability in capabilities:
        # Workaround for Skylark's lack of support for regex. This check should
        # be equivalent to checking:
        #     if re.match("[0-9]+.[0-9]+", capability) == None:
        parts = capability.split(".")
        if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
            auto_configure_fail("Invalid compute capability: %s" % capability)
    return capabilities

def get_cpu_value(repository_ctx):
    """Returns the name of the host operating system.
      Args:
        repository_ctx: The repository context.
      Returns:
        A string containing the name of the host operating system.
      """
    os_name = repository_ctx.os.name.lower()
    if os_name.startswith("mac os"):
        return "Darwin"
    if os_name.find("windows") != -1:
        return "Windows"
    result = repository_ctx.execute(["uname", "-s"])
    return result.stdout.strip()

def _is_windows(repository_ctx):
    """Returns true if the host operating system is windows."""
    return get_cpu_value(repository_ctx) == "Windows"

def lib_name(base_name, cpu_value, version = None, static = False):
    """Constructs the platform-specific name of a library.
      Args:
        base_name: The name of the library, such as "cudart"
        cpu_value: The name of the host operating system.
        version: The version of the library.
        static: True the library is static or False if it is a shared object.
      Returns:
        The platform-specific name of the library.
      """
    version = "" if not version else "." + version
    if cpu_value in ("Linux", "FreeBSD"):
        if static:
            return "lib%s.a" % base_name
        return "lib%s.so%s" % (base_name, version)
    elif cpu_value == "Windows":
        return "%s.lib" % base_name
    elif cpu_value == "Darwin":
        if static:
            return "lib%s.a" % base_name
        return "lib%s%s.dylib" % (base_name, version)
    else:
        auto_configure_fail("Invalid cpu_value: %s" % cpu_value)

def find_lib(repository_ctx, paths, check_soname = True):
    """
      Finds a library among a list of potential paths.
      Args:
        paths: List of paths to inspect.
      Returns:
        Returns the first path in paths that exist.
    """
    objdump = repository_ctx.which("objdump")
    mismatches = []
    for path in [repository_ctx.path(path) for path in paths]:
        if not path.exists:
            continue
        if check_soname and objdump != None and not _is_windows(repository_ctx):
            output = repository_ctx.execute([objdump, "-p", str(path)]).stdout
            output = [line for line in output.splitlines() if "SONAME" in line]
            sonames = [line.strip().split(" ")[-1] for line in output]
            if not any([soname == path.basename for soname in sonames]):
                mismatches.append(str(path))
                continue
        return path
    if mismatches:
        auto_configure_fail(
            "None of the libraries match their SONAME: " + ", ".join(mismatches),
        )
    auto_configure_fail("No library found under: " + ", ".join(paths))

def _find_cuda_lib(
        lib,
        repository_ctx,
        cpu_value,
        basedir,
        version,
        static = False):
    """Finds the given CUDA or cuDNN library on the system.
      Args:
        lib: The name of the library, such as "cudart"
        repository_ctx: The repository context.
        cpu_value: The name of the host operating system.
        basedir: The install directory of CUDA or cuDNN.
        version: The version of the library.
        static: True if static library, False if shared object.
      Returns:
        Returns the path to the library.
      """
    file_name = lib_name(lib, cpu_value, version, static)

    return find_lib(
        repository_ctx,
        ["%s/%s" % (basedir, file_name)],
        check_soname = version and not static,
    )

def _find_libs(repository_ctx, cuda_config):
    """Returns the CUDA and cuDNN libraries on the system.
      Args:
        repository_ctx: The repository context.
        cuda_config: The CUDA config as returned by _get_cuda_config
      Returns:
        Map of library names to structs of filename and path.
      """
    cpu_value = cuda_config.cpu_value
    stub_dir = "" if _is_windows(repository_ctx) else "/stubs"
    return {
        "cuda": _find_cuda_lib(
            "cuda",
            repository_ctx,
            cpu_value,
            cuda_config.config["cuda_library_dir"] + stub_dir,
            None,
        ),
        "cudart": _find_cuda_lib(
            "cudart",
            repository_ctx,
            cpu_value,
            cuda_config.config["cuda_library_dir"],
            cuda_config.cuda_version,
        ),
        "cudart_static": _find_cuda_lib(
            "cudart_static",
            repository_ctx,
            cpu_value,
            cuda_config.config["cuda_library_dir"],
            cuda_config.cuda_version,
            static = True,
        ),
        "cublas": _find_cuda_lib(
            "cublas",
            repository_ctx,
            cpu_value,
            cuda_config.config["cublas_library_dir"],
            cuda_config.cublas_version,
        ),
        "cusolver": _find_cuda_lib(
            "cusolver",
            repository_ctx,
            cpu_value,
            cuda_config.config["cusolver_library_dir"],
            cuda_config.cusolver_version,
        ),
        "curand": _find_cuda_lib(
            "curand",
            repository_ctx,
            cpu_value,
            cuda_config.config["curand_library_dir"],
            cuda_config.curand_version,
        ),
        "cufft": _find_cuda_lib(
            "cufft",
            repository_ctx,
            cpu_value,
            cuda_config.config["cufft_library_dir"],
            cuda_config.cufft_version,
        ),
        "cudnn": _find_cuda_lib(
            "cudnn",
            repository_ctx,
            cpu_value,
            cuda_config.config["cudnn_library_dir"],
            cuda_config.cudnn_version,
        ),
        "cupti": _find_cuda_lib(
            "cupti",
            repository_ctx,
            cpu_value,
            cuda_config.config["cupti_library_dir"],
            cuda_config.cuda_version,
        ),
    }

def _cudart_static_linkopt(cpu_value):
    """Returns additional platform-specific linkopts for cudart."""
    return "" if cpu_value == "Darwin" else "\"-lrt\","

def _get_cuda_config(repository_ctx):
    """Detects and returns information about the CUDA installation on the system.
      Args:
        repository_ctx: The repository context.
      Returns:
        A struct containing the following fields:
          cuda_toolkit_path: The CUDA toolkit installation directory.
          cudnn_install_basedir: The cuDNN installation directory.
          cuda_version: The version of CUDA on the system.
          cudnn_version: The version of cuDNN on the system.
          compute_capabilities: A list of the system's CUDA compute capabilities.
          cpu_value: The name of the host operating system.
      """
    config = find_cuda_config(repository_ctx, ["cuda", "cudnn"])
    cpu_value = get_cpu_value(repository_ctx)
    toolkit_path = config["cuda_toolkit_path"]

    is_windows = _is_windows(repository_ctx)
    cuda_version = config["cuda_version"].split(".")
    cuda_major = cuda_version[0]
    cuda_minor = cuda_version[1]

    cuda_version = ("64_%s%s" if is_windows else "%s.%s") % (cuda_major, cuda_minor)
    cudnn_version = ("64_%s" if is_windows else "%s") % config["cudnn_version"]

    if int(cuda_major) >= 11:
        cublas_version = ("64_%s" if is_windows else "%s") % config["cublas_version"].split(".")[0]
        cusolver_version = ("64_%s" if is_windows else "%s") % config["cusolver_version"].split(".")[0]
        curand_version = ("64_%s" if is_windows else "%s") % config["curand_version"].split(".")[0]
        cufft_version = ("64_%s" if is_windows else "%s") % config["cufft_version"].split(".")[0]
    elif (int(cuda_major), int(cuda_minor)) >= (10, 1):
        # cuda_lib_version is for libraries like cuBLAS, cuFFT, cuSOLVER, etc.
        # It changed from 'x.y' to just 'x' in CUDA 10.1.
        cuda_lib_version = ("64_%s" if is_windows else "%s") % cuda_major
        cublas_version = cuda_lib_version
        cusolver_version = cuda_lib_version
        curand_version = cuda_lib_version
        cufft_version = cuda_lib_version
    else:
        cublas_version = cuda_version
        cusolver_version = cuda_version
        curand_version = cuda_version
        cufft_version = cuda_version

    return struct(
        cuda_toolkit_path = toolkit_path,
        cuda_version = cuda_version,
        cublas_version = cublas_version,
        cusolver_version = cusolver_version,
        curand_version = curand_version,
        cufft_version = cufft_version,
        cudnn_version = cudnn_version,
        compute_capabilities = compute_capabilities(repository_ctx),
        cpu_value = cpu_value,
        config = config,
    )

def _tpl(repository_ctx, tpl, substitutions = {}, out = None):
    if substitutions == None:
        substitutions = {}
    if not out:
        out = tpl.replace(":", "/")
    repository_ctx.template(
        out,
        Label("//build_deps/toolchains/gpu/%s.tpl" % tpl),
        substitutions,
    )

_DUMMY_CROSSTOOL_BUILD_FILE = """
load("//crosstool:error_gpu_disabled.bzl", "error_gpu_disabled")
error_gpu_disabled()
"""

def _create_dummy_repository(repository_ctx):
    cpu_value = get_cpu_value(repository_ctx)

    # Set up BUILD file for cuda/.
    _tpl(
        repository_ctx,
        "cuda:build_defs.bzl",
        {
            "%{cuda_is_configured}": "False",
            "%{cuda_extra_copts}": "[]",
        },
    )
    _tpl(
        repository_ctx,
        "cuda:BUILD",
        {
            "%{cuda_driver_lib}": lib_name("cuda", cpu_value),
            "%{cudart_static_lib}": lib_name(
                "cudart_static",
                cpu_value,
                static = True,
            ),
            "%{cudart_static_linkopt}": _cudart_static_linkopt(cpu_value),
            "%{cudart_lib}": lib_name("cudart", cpu_value),
            "%{cublas_lib}": lib_name("cublas", cpu_value),
            "%{cusolver_lib}": lib_name("cusolver", cpu_value),
            "%{cudnn_lib}": lib_name("cudnn", cpu_value),
            "%{cufft_lib}": lib_name("cufft", cpu_value),
            "%{curand_lib}": lib_name("curand", cpu_value),
            "%{cupti_lib}": lib_name("cupti", cpu_value),
            "%{copy_rules}": "",
            "%{cuda_headers}": "",
        },
    )

    # Create dummy files for the CUDA toolkit since they are still required by
    # tensorflow/core/platform/default/build_config:cuda.
    repository_ctx.file("cuda/cuda/include/cuda.h")
    repository_ctx.file("cuda/cuda/include/cublas.h")
    repository_ctx.file("cuda/cuda/include/cudnn.h")
    repository_ctx.file("cuda/cuda/extras/CUPTI/include/cupti.h")
    repository_ctx.file("cuda/cuda/lib/%s" % lib_name("cuda", cpu_value))
    repository_ctx.file("cuda/cuda/lib/%s" % lib_name("cudart", cpu_value))
    repository_ctx.file(
        "cuda/cuda/lib/%s" % lib_name("cudart_static", cpu_value),
    )
    repository_ctx.file("cuda/cuda/lib/%s" % lib_name("cublas", cpu_value))
    repository_ctx.file("cuda/cuda/lib/%s" % lib_name("cusolver", cpu_value))
    repository_ctx.file("cuda/cuda/lib/%s" % lib_name("cudnn", cpu_value))
    repository_ctx.file("cuda/cuda/lib/%s" % lib_name("curand", cpu_value))
    repository_ctx.file("cuda/cuda/lib/%s" % lib_name("cufft", cpu_value))
    repository_ctx.file("cuda/cuda/lib/%s" % lib_name("cupti", cpu_value))

def _execute(
        repository_ctx,
        cmdline,
        error_msg = None,
        error_details = None,
        empty_stdout_fine = False):
    """Executes an arbitrary shell command.
      Args:
        repository_ctx: the repository_ctx object
        cmdline: list of strings, the command to execute
        error_msg: string, a summary of the error if the command fails
        error_details: string, details about the error or steps to fix it
        empty_stdout_fine: bool, if True, an empty stdout result is fine,
          otherwise it's an error
      Return: the result of repository_ctx.execute(cmdline)
    """
    result = repository_ctx.execute(cmdline)
    if result.stderr or not (empty_stdout_fine or result.stdout):
        auto_configure_fail(
            "\n".join([
                error_msg.strip() if error_msg else "Repository command failed",
                result.stderr.strip(),
                error_details if error_details else "",
            ]),
        )
    return result

def _norm_path(path):
    """Returns a path with '/' and remove the trailing slash."""
    path = path.replace("\\", "/")
    if path[-1] == "/":
        path = path[:-1]
    return path

def make_copy_files_rule(repository_ctx, name, srcs, outs):
    """Returns a rule to copy a set of files."""
    cmds = []

    # Copy files.
    for src, out in zip(srcs, outs):
        cmds.append('cp -f "%s" $(location %s)' % (src, out))
    outs = [('        "%s",' % out) for out in outs]
    return """genrule(
    name = "%s",
    outs = [
%s
    ],
    cmd = \"""%s \""",
)""" % (name, "\n".join(outs), " && ".join(cmds))

def make_copy_dir_rule(repository_ctx, name, src_dir, out_dir):
    """Returns a rule to recursively copy a directory."""
    src_dir = _norm_path(src_dir)
    out_dir = _norm_path(out_dir)
    outs = _read_dir(repository_ctx, src_dir)
    outs = [('        "%s",' % out.replace(src_dir, out_dir)) for out in outs]

    # '@D' already contains the relative path for a single file, see
    # http://docs.bazel.build/versions/master/be/make-variables.html#predefined_genrule_variables
    out_dir = "$(@D)/%s" % out_dir if len(outs) > 1 else "$(@D)"
    return """genrule(
    name = "%s",
    outs = [
%s
    ],
    cmd = \"""cp -rLf "%s/." "%s/" \""",
)""" % (name, "\n".join(outs), src_dir, out_dir)

def _read_dir(repository_ctx, src_dir):
    """Returns a string with all files in a directory.
      Finds all files inside a directory, traversing subfolders and following
      symlinks. The returned string contains the full path of all files
      separated by line breaks.
      """
    if _is_windows(repository_ctx):
        src_dir = src_dir.replace("/", "\\")
        find_result = _execute(
            repository_ctx,
            ["cmd.exe", "/c", "dir", src_dir, "/b", "/s", "/a-d"],
            empty_stdout_fine = True,
        )

        # src_files will be used in genrule.outs where the paths must
        # use forward slashes.
        result = find_result.stdout.replace("\\", "/")
    else:
        find_result = _execute(
            repository_ctx,
            ["find", src_dir, "-follow", "-type", "f"],
            empty_stdout_fine = True,
        )
        result = find_result.stdout
    return sorted(result.splitlines())

def _create_local_cuda_repository(repository_ctx):
    """Creates the repository containing files set up to build with CUDA."""
    cuda_config = _get_cuda_config(repository_ctx)

    cuda_include_path = cuda_config.config["cuda_include_dir"]
    cublas_include_path = cuda_config.config["cublas_include_dir"]
    cudnn_header_dir = cuda_config.config["cudnn_include_dir"]
    cupti_header_dir = cuda_config.config["cupti_include_dir"]
    nvvm_libdevice_dir = cuda_config.config["nvvm_library_dir"]

    # Create genrule to copy files from the installed CUDA toolkit into execroot.
    copy_rules = [
        make_copy_dir_rule(
            repository_ctx,
            name = "cuda-include",
            src_dir = cuda_include_path,
            out_dir = "cuda/include",
        ),
        make_copy_dir_rule(
            repository_ctx,
            name = "cuda-nvvm",
            src_dir = nvvm_libdevice_dir,
            out_dir = "cuda/nvvm/libdevice",
        ),
        make_copy_dir_rule(
            repository_ctx,
            name = "cuda-extras",
            src_dir = cupti_header_dir,
            out_dir = "cuda/extras/CUPTI/include",
        ),
    ]

    copy_rules.append(make_copy_files_rule(
        repository_ctx,
        name = "cublas-include",
        srcs = [
            cublas_include_path + "/cublas.h",
            cublas_include_path + "/cublas_v2.h",
            cublas_include_path + "/cublas_api.h",
        ],
        outs = [
            "cublas/include/cublas.h",
            "cublas/include/cublas_v2.h",
            "cublas/include/cublas_api.h",
        ],
    ))

    cuda_libs = _find_libs(repository_ctx, cuda_config)
    cuda_lib_srcs = []
    cuda_lib_outs = []
    for path in cuda_libs.values():
        cuda_lib_srcs.append(str(path))
        cuda_lib_outs.append("cuda/lib/" + path.basename)
    copy_rules.append(make_copy_files_rule(
        repository_ctx,
        name = "cuda-lib",
        srcs = cuda_lib_srcs,
        outs = cuda_lib_outs,
    ))

    copy_rules.append(make_copy_dir_rule(
        repository_ctx,
        name = "cuda-bin",
        src_dir = cuda_config.cuda_toolkit_path + "/bin",
        out_dir = "cuda/bin",
    ))

    # Copy cudnn.h if cuDNN was not installed to CUDA_TOOLKIT_PATH.
    included_files = _read_dir(repository_ctx, cuda_include_path)
    if not any([file.endswith("cudnn.h") for file in included_files]):
        if [int(x) for x in cuda_config.cudnn_version.split(".")] < [8, 0]:
            cudnn_headers = ["cudnn.h"]
        else:
            cudnn_headers = [
                "cudnn_adv_infer.h",
                "cudnn_adv_train.h",
                "cudnn_cnn_infer.h",
                "cudnn_cnn_train.h",
                "cudnn_ops_infer.h",
                "cudnn_ops_train.h",
                "cudnn.h",
                "cudnn_version.h",
            ]
        cudnn_srcs = []
        cudnn_outs = []
        for header in cudnn_headers:
            cudnn_srcs.append(cudnn_header_dir + "/" + header)
            cudnn_outs.append("cudnn/include/" + header)

        copy_rules.append(make_copy_files_rule(
            repository_ctx,
            name = "cudnn-include",
            srcs = cudnn_srcs,
            outs = cudnn_outs,
        ))
    else:
        copy_rules.append("filegroup(name = 'cudnn-include')\n")

    # Set up BUILD file for cuda/
    _tpl(
        repository_ctx,
        "cuda:build_defs.bzl",
        {
            "%{cuda_is_configured}": "True",
            "%{cuda_extra_copts}": "[]",
        },
    )

    _tpl(
        repository_ctx,
        "cuda:BUILD",
        {
            "%{cuda_driver_lib}": cuda_libs["cuda"].basename,
            "%{cudart_static_lib}": cuda_libs["cudart_static"].basename,
            "%{cudart_static_linkopt}": _cudart_static_linkopt(cuda_config.cpu_value),
            "%{cudart_lib}": cuda_libs["cudart"].basename,
            "%{cublas_lib}": cuda_libs["cublas"].basename,
            "%{cusolver_lib}": cuda_libs["cusolver"].basename,
            "%{cudnn_lib}": cuda_libs["cudnn"].basename,
            "%{cufft_lib}": cuda_libs["cufft"].basename,
            "%{curand_lib}": cuda_libs["curand"].basename,
            "%{cupti_lib}": cuda_libs["cupti"].basename,
            "%{copy_rules}": "\n".join(copy_rules),
            "%{cuda_headers}": (
                '":cuda-include",\n' + '        ":cudnn-include",'
            ),
        },
        "cuda/BUILD",
    )

    # Set up crosstool/
    cc = find_cc(repository_ctx)
    cc_fullpath = cc

    host_compiler_includes = _host_compiler_includes(repository_ctx, cc_fullpath)

    cuda_defines = {}

    # Bazel sets '-B/usr/bin' flag to workaround build errors on RHEL (see
    # https://github.com/bazelbuild/bazel/issues/760).
    # However, this stops our custom clang toolchain from picking the provided
    # LLD linker, so we're only adding '-B/usr/bin' when using non-downloaded
    # toolchain.
    # TODO: when bazel stops adding '-B/usr/bin' by default, remove this
    #       flag from the CROSSTOOL completely (see
    #       https://github.com/bazelbuild/bazel/issues/5634)
    cuda_defines["%{linker_bin_path_flag}"] = 'flag: "-B/usr/bin"'

    cuda_defines["%{host_compiler_path}"] = "clang/bin/crosstool_wrapper_driver_is_not_gcc"
    cuda_defines["%{host_compiler_warnings}"] = ""

    # nvcc has the system include paths built in and will automatically
    # search them; we cannot work around that, so we add the relevant cuda
    # system paths to the allowed compiler specific include paths.
    cuda_defines["%{host_compiler_includes}"] = (
        host_compiler_includes + "\n" + _cuda_include_path(
            repository_ctx,
            cuda_config,
        ) +
        "\n  cxx_builtin_include_directory: \"%s\"" % cupti_header_dir +
        "\n  cxx_builtin_include_directory: \"%s\"" % cudnn_header_dir
    )

    # For gcc, do not canonicalize system header paths; some versions of gcc
    # pick the shortest possible path for system includes when creating the
    # .d file - given that includes that are prefixed with "../" multiple
    # time quickly grow longer than the root of the tree, this can lead to
    # bazel's header check failing.
    cuda_defines["%{extra_no_canonical_prefixes_flags}"] = (
        "flag: \"-fno-canonical-system-headers\""
    )
    nvcc_path = str(
        repository_ctx.path("%s/bin/nvcc%s" % (
            cuda_config.cuda_toolkit_path,
            ".exe" if _is_windows(repository_ctx) else "",
        )),
    )

    builtin_include_directories = []
    for one_line in cuda_defines["%{host_compiler_includes}"].splitlines():
        inc_dir = one_line.split(":")[1][2:-1]
        builtin_include_directories.append(inc_dir)

    _tpl(
        repository_ctx,
        "crosstool:BUILD",
        {
            "%{linker_files}": ":crosstool_wrapper_driver_is_not_gcc",
            "%{cxx_builtin_include_directories}": ",".join(builtin_include_directories),
            "%{win_linker_files}": ":windows_msvc_wrapper_files",
        },
    )
    wrapper_defines = {
        "%{cpu_compiler}": str(cc),
        "%{cuda_version}": cuda_config.cuda_version,
        "%{nvcc_path}": nvcc_path,
        "%{gcc_host_compiler_path}": str(cc),
        "%{cuda_compute_capabilities}": ", ".join(
            ["\"%s\"" % c for c in cuda_config.compute_capabilities],
        ),
        "%{nvcc_tmp_dir}": _get_nvcc_tmp_dir_for_windows(repository_ctx),
    }

    _tpl(
        repository_ctx,
        "crosstool:cc_toolchain_config.bzl",
        wrapper_defines,
    )
    _tpl(
        repository_ctx,
        "crosstool:clang/bin/crosstool_wrapper_driver_is_not_gcc",
        wrapper_defines,
    )

    _tpl(
        repository_ctx,
        "crosstool:windows/msvc_wrapper_for_nvcc.py",
        wrapper_defines,
    )

    _tpl(
        repository_ctx,
        "crosstool:CROSSTOOL",
        cuda_defines.update(_get_win_cuda_defines(repository_ctx)),
        out = "crosstool/CROSSTOOL",
    )

def find_cuda_config(repository_ctx, cuda_libraries):
    """Returns CUDA config dictionary from running find_cuda_config.py"""
    exec_result = repository_ctx.execute([
        _get_python_bin(repository_ctx),
        repository_ctx.path(Label("//build_deps/toolchains/gpu:find_cuda_config.py")),
    ] + cuda_libraries)
    if exec_result.return_code:
        auto_configure_fail("Failed to run find_cuda_config.py: %s" % exec_result.stderr)

    # Parse the dict from stdout.
    return dict([tuple(x.split(": ")) for x in exec_result.stdout.splitlines()])

def _cuda_autoconf_impl(repository_ctx):
    """Implementation of the cuda_autoconf repository rule."""
    if not enable_cuda(repository_ctx):
        _create_dummy_repository(repository_ctx)
    else:
        _create_local_cuda_repository(repository_ctx)

cuda_configure = repository_rule(
    environ = [
        _GCC_HOST_COMPILER_PATH,
        _CLANG_CUDA_COMPILER_PATH,
        "TF_NEED_CUDA",
        "TF_CUDA_CLANG",
        _TF_DOWNLOAD_CLANG,
        _CUDA_TOOLKIT_PATH,
        _CUDNN_INSTALL_PATH,
        _TF_CUDA_VERSION,
        _TF_CUDNN_VERSION,
        _TF_CUDA_COMPUTE_CAPABILITIES,
        "NVVMIR_LIBRARY_DIR",
        _PYTHON_BIN_PATH,
    ],
    implementation = _cuda_autoconf_impl,
)

"""Detects and configures the local CUDA toolchain.
Add the following to your WORKSPACE FILE:
```python
cuda_configure(name = "local_config_cuda")
```
Args:
  name: A unique name for this workspace rule.
"""
