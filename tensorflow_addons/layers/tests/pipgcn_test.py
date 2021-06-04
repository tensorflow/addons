# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for PIPGCN layer."""

import pytest
import numpy as np
import tensorflow as tf
from tensorflow_addons.layers.pipgcn import PIPGCN
from tensorflow_addons.utils import test_utils

@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_output_size():

    batch_size = 32
    graph_mode = 2
    node_in = 70
    edge_in_f = 2
    edge_in_o = 20
    num_outputs = 128

    node = tf.zeros((batch_size, 70), dtype=tf.float32)
    edge = tf.zeros((batch_size, 20, 2), dtype=tf.float32)
    hood = tf.zeros((batch_size, 20, 1), dtype=tf.int32)

    pipgcn = PIPGCN(
        graph_mode=graph_mode,
        node_in=node_in, edge_in_f=edge_in_f, edge_in_o=edge_in_o,
        num_outputs=num_outputs
    )

    output = pipgcn([node, edge, hood])

    assert output.shape[0] == batch_size
    assert output.shape[1] == num_outputs

def test_output_shape():

    batch_size = 32
    graph_mode = 2
    node_in = 70
    edge_in_f = 2
    edge_in_o = 20
    num_outputs = 128

    node = tf.zeros((batch_size, 70), dtype=tf.float32)
    edge = tf.zeros((batch_size, 20, 2), dtype=tf.float32)
    hood = tf.zeros((batch_size, 20, 1), dtype=tf.int32)

    pipgcn = PIPGCN(
        graph_mode=graph_mode,
        node_in=node_in, edge_in_f=edge_in_f, edge_in_o=edge_in_o,
        num_outputs=num_outputs
    )

    output = pipgcn([node, edge, hood])

    assert output.shape[0] == batch_size
    assert output.shape[1] == num_outputs

def test_no_batch():

    batch_size = 32
    graph_mode = 2
    node_in = 70
    edge_in_f = 2
    edge_in_o = 20
    num_outputs = 128

    node = tf.zeros((70), dtype=tf.float32)
    edge = tf.zeros((20, 2), dtype=tf.float32)
    hood = tf.zeros((20, 1), dtype=tf.int32)

    pipgcn = PIPGCN(
        graph_mode=graph_mode,
        node_in=node_in, edge_in_f=edge_in_f, edge_in_o=edge_in_o,
        num_outputs=num_outputs
    )

    output = pipgcn([node, edge, hood])

    assert output.shape[0] == batch_size
    assert output.shape[1] == num_outputs

def test_extra_dims():

    batch_size = 32
    extra_dim = 13
    graph_mode = 2
    node_in = 70
    edge_in_f = 2
    edge_in_o = 20
    num_outputs = 128

    node = tf.zeros((batch_size, extra_dim, 70), dtype=tf.float32)
    edge = tf.zeros((batch_size, extra_dim, 20, 2), dtype=tf.float32)
    hood = tf.zeros((batch_size, extra_dim, 20, 1), dtype=tf.int32)

    pipgcn = PIPGCN(
        graph_mode=graph_mode,
        node_in=node_in, edge_in_f=edge_in_f, edge_in_o=edge_in_o,
        num_outputs=num_outputs
    )

    output = pipgcn([node, edge, hood])

    assert output.shape[0] == batch_size
    assert output.shape[1] == extra_dim
    assert output.shape[2] == num_outputs

def test_compute_output_shape():

    batch_size = 32
    graph_mode = 2
    node_in = 70
    edge_in_f = 2
    edge_in_o = 20
    num_outputs = 128

    pipgcn = PIPGCN(
        graph_mode=graph_mode,
        node_in=node_in, edge_in_f=edge_in_f, edge_in_o=edge_in_o,
        num_outputs=num_outputs
    )

    output = pipgcn.compute_output_shape([batch_size, 70])

    assert output[1] == num_outputs

def test_no_value():

    batch_size = 32
    graph_mode = 2
    node_in = 70
    edge_in_f = 2
    edge_in_o = 20
    num_outputs = 128

    node = tf.zeros((batch_size, 70), dtype=tf.float32)
    edge = tf.zeros((batch_size, 20, 2), dtype=tf.float32)
    hood = tf.zeros((batch_size, 20, 1), dtype=tf.int32)

    pipgcn = PIPGCN(
        graph_mode=graph_mode,
        node_in=node_in, edge_in_f=edge_in_f, edge_in_o=edge_in_o,
        num_outputs=num_outputs
    )

    output = pipgcn([node, edge, hood])

    assert output.shape[0] == batch_size
    assert output.shape[1] == num_outputs

def test_from_to_config():

    batch_size = 32
    graph_mode = 2
    node_in = 70
    edge_in_f = 2
    edge_in_o = 20
    num_outputs = 128

    pipgcn = PIPGCN(
        graph_mode=graph_mode,
        node_in=node_in, edge_in_f=edge_in_f, edge_in_o=edge_in_o,
        num_outputs=num_outputs
    )

    config = pipgcn.get_config()

    new_pipgcn = PIPGCN.from_config(config)

    assert pipgcn.graph_mode == new_pipgcn.graph_mode
    assert pipgcn.node_in == new_pipgcn.node_in
    assert pipgcn.edge_in_f == new_pipgcn.edge_in_f
    assert pipgcn.edge_in_o == new_pipgcn.edge_in_o
    assert pipgcn.num_outputs == new_pipgcn.num_outputs
