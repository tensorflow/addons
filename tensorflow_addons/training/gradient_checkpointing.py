# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of gradient checkpointing methods for memory efficient training."""

from functools import wraps
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.eager import tape as tape_lib


@tf.keras.utils.register_keras_serializable(package="Addons")
def recompute_sequential(f):
    r"""Decorator function that enables recomputing the intermediate outputs (activations) of a sequential keras model in eager mode.
    This allows training large models on limited memory settings but at the cost of increased training time.
    Args:
        f: Keras sequential model to be wrapped
    Returns:
        A function 'inner' that wraps a function 'grad'."""

    @wraps(f)
    def inner(*args, _watch_vars=None, **kwargs):
        r"""Performs a forward pass without storing the activations """
        if _watch_vars is None:
            _watch_vars = []
        tensor_watches = [tf.convert_to_tensor(x) for x in _watch_vars]

        with tape_lib.stop_recording():
            result = f(*args, **kwargs)
            flat_result = nest.flatten(result)
            flat_result = [tf.identity(x) for x in flat_result]
            output = nest.pack_sequence_as(result, flat_result)

        model, x = args
        x = tf.convert_to_tensor(x)

        def grad(*grads_output):
            r"""Performs the backward pass while recomputing the forward pass activations for each layer. """
            grads = []
            for idx_forward in range(len(model.layers)):
                idx_back = len(model.layers) - idx_forward - 1
                back_layer = model.layers[idx_back]
                unique_vars = []
                # Note - CI checks insist on using 'ref' instead of 'experimental ref'. But using 'ref' with nightly
                # builds seems to consume more memory compared to 'experiemntal_ref'.
                if back_layer.trainable_variables:
                    unique_vars = [
                        v.deref()
                        for v in set(v.ref() for v in back_layer.trainable_variables)
                    ]
                prev_output = x
                for idx_layer in range(idx_back):
                    prev_output = model.layers[idx_layer](prev_output)
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(back_layer.trainable_variables)
                    tape.watch(prev_output)
                    recomputed_output = back_layer(prev_output)
                    # identity necessary for grad propagation across 'dead' layers
                    recomputed_output = [tf.identity(x) for x in recomputed_output]
                    recomputed_output = tf.convert_to_tensor(recomputed_output)
                    prev_output = nest.flatten(prev_output)
                    sources = prev_output + unique_vars
                grads_intermediate = tape.gradient(
                    recomputed_output, sources, output_gradients=grads_output
                )
                grads_output = grads_intermediate[: len(prev_output)]
                grads_vars = grads_intermediate[len(prev_output) :]
                grads.extend(grads_vars[::-1])
                del tape
            return grads[::-1]

        tape_lib.record_operation(str(f), flat_result, tensor_watches, grad)

        return output

    return inner
