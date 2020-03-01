import glob
import os
from pathlib import Path

import tensorflow as tf

from tensorflow_addons.utils.resource_loader import get_project_root


def register_all(keras_objects: bool = True, custom_kernels: bool = True) -> None:
    if keras_objects:
        register_keras_objects()
    if custom_kernels:
        register_custom_kernels()


def register_keras_objects() -> None:
    # TODO: once layer_test is replaced by a public API
    # and we can used unregistered objects with it
    # we can remove all decorators.
    # And register Keras objects here.
    pass


def register_custom_kernels() -> None:
    all_shared_objects = _get_all_shared_objects()
    if not all_shared_objects:
        raise FileNotFoundError(
            "No shared objects files were found in the custom ops "
            "directory in Tensorflow Addons, check your installation again,"
            "or, if you don't need custom ops, call `tfa.register_all(custom_kernels=False)`"
            " instead."
        )
    try:
        for shared_object in all_shared_objects:
            tf.load_op_library(shared_object)
    except tf.errors.NotFoundError as e:
        raise RuntimeError(
            "One of the shared objects ({}) could not be loaded. This may be "
            "due to a number of reasons (incompatible TensorFlow version, buiding from "
            "source with different flags, broken install of TensorFlow Addons...). If you"
            "wanted to register the shared objects because you needed them when loading your "
            "model, you should fix your install of TensorFlow Addons. If you don't "
            "use custom ops in your model, you can skip registering custom ops with "
            "`tfa.register_all(custom_kernels=False)`".format(shared_object)
        ) from e


def _get_all_shared_objects():
    custom_ops_dir = os.path.join(get_project_root(), "custom_ops")
    all_shared_objects = glob.glob(custom_ops_dir + "/**/*.so", recursive=True)
    all_shared_objects = [x for x in all_shared_objects if Path(x).is_file()]
    return list(all_shared_objects)
