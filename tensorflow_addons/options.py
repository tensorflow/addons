import os
import platform
import warnings
import traceback

try:
    _TF_ADDONS_PY_OPS = bool(int(os.environ["TF_ADDONS_PY_OPS"]))
except KeyError:
    if platform.system() == "Linux":
        _TF_ADDONS_PY_OPS = False
    else:
        _TF_ADDONS_PY_OPS = True

FALLBACK_WARNING_TEMPLATE = """{}

The {} C++/CUDA custom op could not be loaded.
For this reason, Addons will fallback to an implementation written
in Python with public TensorFlow ops. There worst you might experience with
this is a moderate slowdown on GPU. There can be multiple
reason for this loading error, one of them may be an ABI incompatibility between
the TensorFlow installed on your system and the TensorFlow used to compile
TensorFlow Addons' custom ops. The stacktrace generated when loading the
shared object file was displayed above.

If you want this warning to disappear, either make sure the TensorFlow installed
is compatible with this version of Addons, or tell TensorFlow Addons to
prefer using Python implementations and not custom C++/CUDA ones. You can do that
by changing the _TF_ADDONS_PY_OPS flag
either with the environment variable:
```bash
TF_ADDONS_PY_OPS=1 python my_script.py
```
or in your code, after your imports:
```python
import tensorflow_addons as tfa
import ...
import ...

tfa.options.disable_custom_kernel()
```
"""


def warn_fallback(op_name):
    warning_msg = FALLBACK_WARNING_TEMPLATE.format(traceback.format_exc(), op_name)
    warnings.warn(warning_msg, RuntimeWarning)
    global _TF_ADDONS_PY_OPS
    _TF_ADDONS_PY_OPS = True


def enable_custom_kernel():
    """Prefer custom kernel over pure python kernel.

    Enable using custom CUDA/C++ Kernel instead of pure python kernel. Use this instead of
    directly accessing the global variable."""
    global _TF_ADDONS_PY_OPS
    _TF_ADDONS_PY_OPS = False
    pass


def disable_custom_kernel():
    """Prefer python kernel over custom kernel.

    Disable using custom CUDA/C++ Kernel instead of pure python kernel. Use this instead of
    directly accessing the global variable."""
    global _TF_ADDONS_PY_OPS
    _TF_ADDONS_PY_OPS = True
    pass


def is_custom_kernel_disabled():
    """Returns whether pure python kernel is preferred or not."""
    return _TF_ADDONS_PY_OPS
