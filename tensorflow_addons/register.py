import inspect
import glob
import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable
import warnings
from tensorflow_addons import (
    activations,
    callbacks,
    image,
    layers,
    losses,
    metrics,
    optimizers,
    rnn,
    seq2seq,
)
from tensorflow_addons.utils.resource_loader import get_project_root
import os

SUBMODULES = [
    activations,
    callbacks,
    image,
    layers,
    losses,
    metrics,
    optimizers,
    rnn,
    seq2seq,
]

already_registered = False


def register_all(keras_objects: bool = True, custom_kernels: bool = True) -> None:
    if keras_objects:
        register_keras_objects()
    if custom_kernels:
        register_custom_kernels()


def register_keras_objects() -> None:
    global already_registered
    if already_registered:
        warnings.warn(
            "Tensorflow Addons' functions and classes are already "
            "registered in the Keras custom objects dictionary.",
            UserWarning,
        )
    for module in SUBMODULES:
        for attribute in _get_attributes(module):
            if inspect.isclass(attribute) or inspect.isfunction(attribute):
                register_keras_serializable(package="Addons")(attribute)

    already_registered = True


def register_custom_kernels() -> None:
    custom_ops_dir = os.path.join(get_project_root(), "custom_ops")
    all_shared_objects = glob.glob(custom_ops_dir + "/**/*.so", recursive=True)
    for shared_object in all_shared_objects:
        tf.load_op_library(shared_object)


def _get_attributes(module):
    for attr_name in dir(module):
        if attr_name.startswith("_"):
            continue
        attr = getattr(module, attr_name)
        yield attr
