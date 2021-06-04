# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Implements PIPGCN Layer."""

import tensorflow as tf
from typeguard import typechecked


@tf.keras.utils.register_keras_serializable(package="Addons")
class PIPGCN(tf.keras.layers.Layer):
    """ PIPGCN with three mode
    See [PIPGCN: Protein Interface Prediction using Graph Convolutional Networks](https://mountainscholar.org/handle/10217/185661)
    Args:
        graph_mode:
            0: node average
            1: node edge average
            2: order dependent average
        node_in: number of node feature
        edge_in_f: number of node edge
        edge_in_o: number of closest residues
        num_outputs: feature dimension of the output tensor
    """

    @typechecked
    def __init__(self,
                 graph_mode: int,
                 node_in: int, edge_in_f: int, edge_in_o: int,
                 num_outputs: int, **kwargs):

        super().__init__(**kwargs)

        if(graph_mode < 0 or graph_mode > 2):
            raise ValueError("`graph_mode` must be one of 0, 1, or 2: %i" % graph_mode)
        if(num_outputs <= 0):
            raise ValueError("`num_outputs` must be greater than 1: %i" % num_outputs)

        self.graph_mode = graph_mode
        self.node_in = node_in
        self.edge_in_f = edge_in_f
        self.edge_in_o = edge_in_o
        self.num_outputs = num_outputs

    def build(self, input_shape):
        """Keras build method."""

        self.w_node_c = self.add_weight(
            name="node_c",
            shape=(self.node_in, self.num_outputs),
            initializer=tf.keras.initializers.GlorotNormal(),
            trainable=True,
        )

        if(self.graph_mode == 0):
            pass
        elif(self.graph_mode >= 1):
            self.w_node_n = self.add_weight(
                name="node_n",
                shape=(self.node_in, self.num_outputs),
                initializer=tf.keras.initializers.GlorotNormal(),
                trainable=True,
            )

            if(self.graph_mode == 1):
                self.w_edge = self.add_weight(
                    name="edge",
                    shape=(self.edge_in_f, self.num_outputs),
                    initializer=tf.keras.initializers.GlorotNormal(),
                    trainable=True,
                )
            elif(self.graph_mode == 2):
                self.w_edge_order = self.add_weight(
                    name="edge_o",
                    shape=(self.edge_in_o, self.num_outputs),
                    initializer=tf.keras.initializers.GlorotNormal(),
                    trainable=True,
                )
                self.w_edge_feat = self.add_weight(
                    name="edge_f",
                    shape=(self.edge_in_f, self.num_outputs),
                    initializer=tf.keras.initializers.GlorotNormal(),
                    trainable=True,
                )

        super(PIPGCN, self).build(input_shape)

    def call(self, inputs):

        if(self.graph_mode == 0):
            node = inputs['node']

            """ -=-=-= Term 1: node aggregation =-=-=- """
            term1 = tf.linalg.matmul(node, self.w_node_c)

            y = term1
        elif(self.graph_mode == 1):
            node = inputs['node']
            edge = inputs['edge']
            hood = tf.squeeze(inputs['hood'], axis=2)
            hood_in = tf.expand_dims(tf.math.count_nonzero(hood + 1, axis=1, dtype=tf.float32), -1)

            """ -=-=-= Term 1: node aggregation =-=-=- """
            term1 = tf.linalg.matmul(node, self.w_node_c)

            """ -=-=-= Term 2: edge aggregation =-=-=- """
            wn = tf.linalg.matmul(node, self.w_node_n)
            we = tf.linalg.matmul(edge, self.w_edge)
            gather_n = tf.gather(wn, hood)
            node_avg = tf.reduce_sum(gather_n, 1)
            edge_avg = tf.reduce_sum(we, 1)
            numerator = node_avg + edge_avg
            denominator = tf.maximum(hood_in, tf.ones_like(hood_in))
            term2 = tf.math.divide(numerator, denominator)

            y = term1 + term2
        elif(self.graph_mode == 2):
            node = inputs['node']
            edge = inputs['edge']
            hood = tf.squeeze(inputs['hood'], axis=2)
            hood_in = tf.expand_dims(tf.math.count_nonzero(hood + 1, axis=1, dtype=tf.float32), -1)

            """ -=-=-= Term 1: node aggregation =-=-=- """
            term1 = tf.linalg.matmul(node, self.w_node_c)

            """ -=-=-= Term 2: edge aggregation =-=-=- """
            wn = tf.linalg.matmul(node, self.w_node_n)
            we_o = tf.linalg.matmul(tf.transpose(edge, perm=[0, 2, 1]), self.w_edge_order)
            we_f = tf.linalg.matmul(edge, self.w_edge_feat)
            gather_n = tf.gather(wn, hood)
            node_avg = tf.reduce_sum(gather_n, 1)
            edge_order = tf.reduce_sum(we_o, 1) + tf.reduce_sum(we_f, 1)
            numerator = node_avg + edge_order
            denominator = tf.maximum(hood_in, tf.ones_like(hood_in))
            term2 = tf.math.divide(numerator, denominator)

            y = term1 + term2

        return y

    def compute_output_shape(self, input_shape):

        input_shape = tf.TensorShape(input_shape).as_list()
        return tf.TensorShape([input_shape[0], self.num_outputs])

    def get_config(self):

        config = {"num_outputs": self.num_outputs}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
