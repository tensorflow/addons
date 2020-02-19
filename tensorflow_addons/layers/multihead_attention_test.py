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
# =============================================================================

import numpy as np
import tensorflow as tf

from tensorflow_addons.layers.multihead_attention import MultiHeadAttention
from tensorflow_addons.utils import test_utils


@test_utils.run_all_in_graph_and_eager_modes
class MultiHeadAttentionTest(tf.test.TestCase):
    def test_output_shape(self):
        batch_size = 10
        num_heads = 8
        head_size = 12

        q = tf.random.uniform((batch_size, 5, 9), dtype=np.float32)
        k = tf.random.uniform((batch_size, 7, 11), dtype=np.float32)
        v = tf.random.uniform((batch_size, 7, 13), dtype=np.float32)

        mha = MultiHeadAttention(head_size=head_size, num_heads=num_heads)

        output = mha([q, k, v])

        self.assertEqual(output.shape[0], batch_size)
        self.assertEqual(output.shape[1], q.shape[1])
        self.assertEqual(output.shape[2], v.shape[2])

    def test_attention_coefficients_shape(self):
        batch_size = 10
        num_heads = 8
        head_size = 12

        q = tf.random.uniform((batch_size, 5, 9), dtype=np.float32)
        k = tf.random.uniform((batch_size, 7, 11), dtype=np.float32)
        v = tf.random.uniform((batch_size, 7, 13), dtype=np.float32)

        mha = MultiHeadAttention(
            head_size=head_size, num_heads=num_heads, return_attn_coef=True
        )

        output, attn_coef = mha([q, k, v])

        self.assertEqual(attn_coef.shape[0], batch_size)
        self.assertEqual(attn_coef.shape[1], num_heads)
        self.assertEqual(attn_coef.shape[2], q.shape[1])
        self.assertEqual(attn_coef.shape[3], k.shape[1])

        self.assertEqual(output.shape[1], q.shape[1])
        self.assertEqual(output.shape[2], v.shape[2])

    def test_mask(self):
        batch_size = 10
        num_heads = 8
        head_size = 12

        q = tf.random.uniform((batch_size, 5, 9), dtype=np.float32)
        k = tf.random.uniform((batch_size, 7, 11), dtype=np.float32)
        v = tf.random.uniform((batch_size, 7, 13), dtype=np.float32)
        mask = tf.random.uniform((batch_size, num_heads, 5, 7), dtype=np.float32) > 0.1

        mha = MultiHeadAttention(
            head_size=head_size, num_heads=num_heads, return_attn_coef=True
        )

        output, attn_coef = mha([q, k, v, mask])

        self.assertEqual(attn_coef.shape[0], batch_size)
        self.assertEqual(attn_coef.shape[1], num_heads)
        self.assertEqual(attn_coef.shape[2], q.shape[1])
        self.assertEqual(attn_coef.shape[3], k.shape[1])

        self.assertEqual(output.shape[0], batch_size)
        self.assertEqual(output.shape[1], q.shape[1])
        self.assertEqual(output.shape[2], v.shape[2])

        if tf.executing_eagerly():
            attn_coef = attn_coef.numpy()
            mask = mask.numpy()

            self.assertTrue(((attn_coef != 0) == mask).all())

    def test_mask_no_batch(self):
        batch_size = 10
        num_heads = 8
        head_size = 12

        q = tf.random.uniform((batch_size, 5, 9), dtype=np.float32)
        k = tf.random.uniform((batch_size, 7, 11), dtype=np.float32)
        v = tf.random.uniform((batch_size, 7, 13), dtype=np.float32)
        mask = tf.random.uniform((5, 7), dtype=np.float32) > 0.1

        mha = MultiHeadAttention(
            head_size=head_size, num_heads=num_heads, return_attn_coef=True
        )

        output, attn_coef = mha([q, k, v, mask])

        self.assertEqual(output.shape[0], batch_size)
        self.assertEqual(output.shape[1], q.shape[1])
        self.assertEqual(output.shape[2], v.shape[2])

        if tf.executing_eagerly():
            attn_coef = attn_coef.numpy()
            mask = mask.numpy()

            self.assertTrue(((attn_coef != 0) == mask).all())


if __name__ == "__main__":
    tf.test.main()
