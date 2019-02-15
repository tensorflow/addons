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
"""Tests for lifted loss."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.ops import convert_to_tensor
from tensorflow_addons.losses.python import lifted
from tensorflow_addons.losses.python import metric_learning

@test_util.run_all_in_graph_and_eager_modes
class LiftedStructLossTest(test.TestCase):
    def testLiftedStruct(self):
        with self.cached_session():
            num_data = 10
            feat_dim = 6
            margin = 1.0
            num_classes = 4

            embedding = np.random.rand(num_data, feat_dim).astype(np.float32)
            labels = np.random.randint(0, num_classes, size=(num_data)).astype(np.float32)
            # Reshape labels to compute adjacency matrix.
            labels_reshaped = np.reshape(labels, (labels.shape[0], 1))

            # Compute the loss in NP
            adjacency = np.equal(labels_reshaped, labels_reshaped.T)
            pdist_matrix = metric_learning.pairwise_distance_np(embedding)
            loss_np = 0.0
            num_constraints = 0.0
            for i in range(num_data):
                for j in range(num_data):
                    if adjacency[i][j] > 0.0 and i != j:
                        d_pos = pdist_matrix[i][j]
                        negs = []
                        for k in range(num_data):
                            if not adjacency[i][k]:
                                negs.append(margin - pdist_matrix[i][k])
                        for l in range(num_data):
                            if not adjacency[j][l]:
                                negs.append(margin - pdist_matrix[j][l])

                    negs = np.array(negs)
                    max_elem = np.max(negs)
                    negs -= max_elem
                    negs = np.exp(negs)
                    soft_maximum = np.log(np.sum(negs)) + max_elem

                    num_constraints += 1.0
                    this_loss = max(soft_maximum + d_pos, 0)
                    loss_np += this_loss * this_loss

            loss_np = loss_np / num_constraints / 2.0

            # Compute the loss in TF
            loss_tf = lifted.lifted_struct_loss(
                labels=convert_to_tensor(labels),
                embeddings=convert_to_tensor(embedding),
                margin=margin)
            loss_tf = loss_tf.eval()
            self.assertAllClose(loss_np, loss_tf)

if __name__ == '__main__':
    test.main()