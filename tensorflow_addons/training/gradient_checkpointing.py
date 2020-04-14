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
import numpy as np
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
    def inner(*args, _watch_vars=None, num_checkpoints=0, **kwargs):
        r"""Performs a forward pass while storing only the checkpoint activations """
        if _watch_vars is None:
            _watch_vars = []
        tensor_watches = [tf.convert_to_tensor(x) for x in _watch_vars]
        model, x = args
        # Dictionary to cache the desired activations during forward pass
        saved_tensors = {}
        # index -1 represents the inputs x
        idx_ckpt = np.array([-1])
        num_layers = len(model.layers)
        # Perform checkpointing. Naive scheme - just distribute checkpoints uniformly across the layers.
        if num_checkpoints:
            if num_checkpoints >= num_layers:
                raise ValueError(
                    "The number of checkpoints is {} and should be less than number of"
                    "layers in the model, which is {} .".format(
                        num_checkpoints, num_layers
                    )
                )
            idx_start, idx_end = 0, num_layers - 1
            # Use offset to avoid checkpointing the start and end layers of the model
            offset = idx_end // num_checkpoints
            start, end = (idx_start + offset) // 2, (idx_end - offset + idx_end) // 2
            idx_tmp = np.linspace(start, end, num_checkpoints, dtype=np.uint32)
            idx_ckpt = np.append(idx_ckpt, idx_tmp).tolist()

        x = tf.convert_to_tensor(x)
        with tape_lib.stop_recording():
            # perform forward pass while caching checkpoint layer outputs
            result = x
            saved_tensors[-1] = result
            for idx_layer in range(num_layers):
                result = model.layers[idx_layer](result)
                if idx_layer in idx_ckpt:
                    saved_tensors[idx_layer] = result
            flat_result = nest.flatten(result)
            flat_result = [tf.identity(x) for x in flat_result]
            output = nest.pack_sequence_as(result, flat_result)

        def grad(*grads_output):
            r"""Performs the backward pass while recomputing the forward pass activations for each layer. """
            grads = []
            for idx_forward in range(len(model.layers)):
                idx_back = len(model.layers) - idx_forward - 1
                back_layer = model.layers[idx_back]
                idx_last_ckpt = idx_ckpt[-1]
                if idx_back <= idx_last_ckpt:
                    idx_ckpt.pop()
                    idx_last_ckpt = idx_ckpt[-1]
                prev_output = saved_tensors[idx_last_ckpt]
                for idx_layer in range(idx_last_ckpt + 1, idx_back):
                    prev_output = model.layers[idx_layer](prev_output)
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(back_layer.trainable_variables)
                    tape.watch(prev_output)
                    recomputed_output = back_layer(prev_output)
                    # identity necessary for grad propagation across 'dead' layers
                    recomputed_output = [tf.identity(x) for x in recomputed_output]
                    recomputed_output = tf.convert_to_tensor(recomputed_output)
                    prev_output = nest.flatten(prev_output)
                    sources = prev_output + back_layer.trainable_variables
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
