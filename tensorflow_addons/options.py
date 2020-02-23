import os
import platform

try:
    TF_ADDONS_PY_OPS = bool(int(os.environ["TF_ADDONS_PY_OPS"]))
except KeyError:
    if platform.system() == "Linux":
        TF_ADDONS_PY_OPS = False
    else:
        TF_ADDONS_PY_OPS = True
