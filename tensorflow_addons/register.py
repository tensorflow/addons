import glob
from pathlib import Path

import tensorflow as tf

from tensorflow_addons.utils.resource_loader import get_path_to_datafile


def register_all(keras_objects: bool = True, custom_kernels: bool = True) -> None:
    """Register TensorFlow Addons' objects in TensorFlow global dictionaries.

    When loading a Keras model that has a TF Addons' function, it is needed
    for this function to be known by the Keras deserialization process.

    There are two ways to do this, either do

    ```python
    tf.keras.models.load_model(
        "my_model.tf",
        custom_objects={"LAMB": tfa.image.optimizer.LAMB}
    )
    ```

    or you can do:
    ```python
    tfa.register_all()
    tf.keras.models.load_model("my_model.tf")
    ```

    If the model contains custom ops (compiled ops) of TensorFlow Addons,
    and the graph is loaded with `tf.saved_model.load`, then custom ops need
    to be registered before to avoid an error of the type:

    ```
    tensorflow.python.framework.errors_impl.NotFoundError: Op type not registered
    '...' in binary running on ... Make sure the Op and Kernel are
    registered in the binary running in this process.
    ```

    In this case, the only way to make sure that the ops are registered is to call
    this function:

    ```python
    tfa.register_all()
    tf.saved_model.load("my_model.tf")
    ```

    Note that you can call this function multiple times in the same process,
    it only has an effect the first time. Afterward, it's just a no-op.

    Args:
        keras_objects: boolean, `True` by default. If `True`, register all
            Keras objects
            with `tf.keras.utils.register_keras_serializable(package="Addons")`
            If set to False, doesn't register any Keras objects
            of Addons in TensorFlow.
        custom_kernels: boolean, `True` by default. If `True`, loads all
            custom kernels of TensorFlow Addons with
            `tf.load_op_library("path/to/so/file.so")`. Loading the SO files
            register them automatically. If `False` doesn't load and register
            the shared objects files. Not that it might be useful to turn it off
            if your installation of Addons doesn't work well with custom ops.
    Returns:
        None
    """
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
            "directory in Tensorflow Addons, check your installation again, "
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
            "source with different flags, broken install of TensorFlow Addons...). If you "
            "wanted to register the shared objects because you needed them when loading your "
            "model, you should fix your install of TensorFlow Addons. If you don't "
            "use custom ops in your model, you can skip registering custom ops with "
            "`tfa.register_all(custom_kernels=False)`".format(shared_object)
        ) from e


def _get_all_shared_objects():
    custom_ops_dir = get_path_to_datafile("custom_ops", is_so=True)
    all_shared_objects = glob.glob(custom_ops_dir + "/**/*.so", recursive=True)
    all_shared_objects = [x for x in all_shared_objects if Path(x).is_file()]
    return all_shared_objects
