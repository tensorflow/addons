# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

import tensorflow as tf
import numpy as np

from tensorflow_addons.activations import sparsemax
from tensorflow_addons.losses import sparsemax_loss, SparsemaxLoss
from tensorflow_addons.utils import test_utils

test_obs = 17


def _np_sparsemax(z):
    z = z - np.mean(z, axis=1)[:, np.newaxis]

    # sort z
    z_sorted = np.sort(z, axis=1)[:, ::-1]

    # calculate k(z)
    z_cumsum = np.cumsum(z_sorted, axis=1)
    k = np.arange(1, z.shape[1] + 1)
    z_check = 1 + k * z_sorted > z_cumsum
    # use argmax to get the index by row as .nonzero() doesn't
    # take an axis argument. np.argmax return the first index, but the last
    # index is required here, use np.flip to get the last index and
    # `z.shape[axis]` to compensate for np.flip afterwards.
    k_z = z.shape[1] - np.argmax(z_check[:, ::-1], axis=1)

    # calculate tau(z)
    tau_sum = z_cumsum[np.arange(0, z.shape[0]), k_z - 1]
    tau_z = ((tau_sum - 1) / k_z).reshape(-1, 1)

    # calculate p
    return np.maximum(0, z - tau_z)


def _np_sparsemax_loss(z, q):
    z = z - np.mean(z, axis=1)[:, np.newaxis]

    # Calculate q^T * z
    z_k = np.sum(q * z, axis=1)

    # calculate sum over S(z)
    p = _np_sparsemax(z)
    s = p > 0
    # z_i^2 - tau(z)^2 = p_i (2 * z_i - p_i) for i \in S(z)
    S_sum = np.sum(s * p * (2 * z - p), axis=1)

    # because q is binary, sum([q_1^2, q_2^2, ...]) is just sum(q)
    q_norm = np.sum(q, axis=1)

    return 0.5 * S_sum + 0.5 * q_norm - z_k


@test_utils.run_all_with_types(["float32", "float64"])
@test_utils.run_all_in_graph_and_eager_modes
class SparsemaxTest(tf.test.TestCase):
    def _tf_sparsemax(self, z, dtype):
        tf_sparsemax_op = sparsemax(z.astype(dtype))
        tf_sparsemax_out = self.evaluate(tf_sparsemax_op)

        return tf_sparsemax_op, tf_sparsemax_out

    def _tf_sparsemax_loss(self, z, q, dtype):
        z = z.astype(dtype)
        q = q.astype(dtype)

        tf_sparsemax_op = sparsemax(z)
        tf_loss_op = sparsemax_loss(z, tf_sparsemax_op, q)
        tf_loss_out = self.evaluate(tf_loss_op)

        return tf_loss_op, tf_loss_out

    def test_sparsemax_loss_constructor_aginst_numpy(self, dtype=None):
        """check sparsemax-loss construcor against numpy."""
        random = np.random.RandomState(1)

        z = random.uniform(low=-3, high=3, size=(test_obs, 10))
        q = np.zeros((test_obs, 10))
        q[np.arange(0, test_obs), random.randint(0, 10, size=test_obs)] = 1

        loss_object = SparsemaxLoss()
        tf_loss_op = loss_object(q, z)
        tf_loss_out = self.evaluate(tf_loss_op)
        np_loss = np.mean(_np_sparsemax_loss(z, q).astype(dtype))

        self.assertAllCloseAccordingToType(np_loss, tf_loss_out)
        self.assertShapeEqual(np_loss, tf_loss_op)

    def test_sparsemax_loss_constructor_not_from_logits(self, dtype=None):
        """check sparsemax-loss construcor throws when from_logits=True."""
        self.assertRaises(ValueError, lambda: SparsemaxLoss(from_logits=False))

    def test_sparsemax_loss_against_numpy(self, dtype=None):
        """check sparsemax-loss kernel against numpy."""
        random = np.random.RandomState(1)

        z = random.uniform(low=-3, high=3, size=(test_obs, 10))
        q = np.zeros((test_obs, 10))
        q[np.arange(0, test_obs), random.randint(0, 10, size=test_obs)] = 1

        tf_loss_op, tf_loss_out = self._tf_sparsemax_loss(z, q, dtype)
        np_loss = _np_sparsemax_loss(z, q).astype(dtype)

        self.assertAllCloseAccordingToType(np_loss, tf_loss_out)
        self.assertShapeEqual(np_loss, tf_loss_op)

    def test_sparsemax_loss_of_nan(self, dtype=None):
        """check sparsemax-loss transfers nan."""
        random = np.random.RandomState(2)

        q = np.asarray([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
        z_nan = np.asarray(
            [[0, np.nan, 0], [0, np.nan, np.nan], [np.nan, np.nan, np.nan]]
        ).astype(dtype)

        _, tf_loss_nan = self._tf_sparsemax_loss(z_nan, q, dtype)
        self.assertAllEqual([np.nan, np.nan, np.nan], tf_loss_nan)

    def test_sparsemax_loss_of_inf(self, dtype=None):
        """check sparsemax-loss is infinity safe."""
        random = np.random.RandomState(3)

        q = np.asarray([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]])
        z_neg = np.asarray(
            [
                [0, -np.inf, 0],
                [0, -np.inf, -np.inf],
                [-np.inf, -np.inf, 0],
                [-np.inf, -np.inf, -np.inf],
            ]
        ).astype(dtype)
        z_pos = np.asarray(
            [
                [0, np.inf, 0],
                [0, np.inf, np.inf],
                [np.inf, np.inf, 0],
                [np.inf, np.inf, np.inf],
            ]
        ).astype(dtype)
        z_mix = np.asarray(
            [
                [0, np.inf, 0],
                [0, np.inf, -np.inf],
                [-np.inf, np.inf, 0],
                [-np.inf, np.inf, -np.inf],
            ]
        ).astype(dtype)

        _, tf_loss_neg = self._tf_sparsemax_loss(z_neg, q, dtype)
        self.assertAllEqual([0.25, np.inf, 0, np.nan], tf_loss_neg)

        _, tf_loss_pos = self._tf_sparsemax_loss(z_pos, q, dtype)
        self.assertAllEqual([np.nan, np.nan, np.nan, np.nan], tf_loss_pos)

        _, tf_loss_mix = self._tf_sparsemax_loss(z_mix, q, dtype)
        self.assertAllEqual([np.nan, np.nan, np.nan, np.nan], tf_loss_mix)

    def test_constant_add(self, dtype=None):
        """check sparsemax-loss proposition 3."""
        random = np.random.RandomState(4)

        z = random.uniform(low=-3, high=3, size=(test_obs, 10))
        c = random.uniform(low=-3, high=3, size=(test_obs, 1))
        q = np.zeros((test_obs, 10))
        q[np.arange(0, test_obs), np.random.randint(0, 10, size=test_obs)] = 1

        _, tf_loss_zpc = self._tf_sparsemax_loss(z + c, q, dtype)
        _, tf_loss_z = self._tf_sparsemax_loss(z, q, dtype)

        self.assertAllCloseAccordingToType(
            tf_loss_zpc, tf_loss_z, float_atol=5e-6, float_rtol=5e-6
        )

    def test_sparsemax_loss_positive(self, dtype=None):
        """check sparsemax-loss proposition 4."""
        random = np.random.RandomState(5)

        z = random.uniform(low=-3, high=3, size=(test_obs, 10))
        q = np.zeros((test_obs, 10))
        q[np.arange(0, test_obs), random.randint(0, 10, size=test_obs)] = 1

        tf_loss_op, tf_loss_out = self._tf_sparsemax_loss(z, q, dtype)

        self.assertAllCloseAccordingToType(np.abs(tf_loss_out), tf_loss_out)
        self.assertShapeEqual(np.zeros(test_obs), tf_loss_op)

    def test_sparsemax_loss_zero(self, dtype=None):
        """check sparsemax-loss proposition 5."""
        random = np.random.RandomState(6)

        # construct z and q, such that z_k >= 1 + max_{j!=k} z_k holds for
        # delta_0 = 1.
        z = random.uniform(low=-3, high=3, size=(test_obs, 10))
        z[:, 0] = np.max(z, axis=1) + 1.05

        q = np.zeros((test_obs, 10))
        q[:, 0] = 1

        tf_loss_op, tf_loss_out = self._tf_sparsemax_loss(z, q, dtype)
        tf_sparsemax_op, tf_sparsemax_out = self._tf_sparsemax(z, dtype)

        self.assertAllCloseAccordingToType(np.zeros(test_obs), tf_loss_out)
        self.assertShapeEqual(np.zeros(test_obs), tf_loss_op)

        self.assertAllCloseAccordingToType(q, tf_sparsemax_out)
        self.assertShapeEqual(q, tf_sparsemax_op)

    def test_gradient_against_estimate(self, dtype=None):
        """check sparsemax-loss Rop, against estimated-loss Rop."""
        random = np.random.RandomState(7)

        # sparsemax is not a smooth function so gradient estimation is only
        # possible for float64.
        if dtype != "float64":
            return

        z = random.uniform(low=-3, high=3, size=(test_obs, 10)).astype(dtype)
        q = np.zeros((test_obs, 10)).astype(dtype)
        q[np.arange(0, test_obs), np.random.randint(0, 10, size=test_obs)] = 1

        (jacob_sym,), (jacob_num,) = tf.test.compute_gradient(
            lambda logits: sparsemax_loss(logits, sparsemax(logits), q), [z]
        )
        self.assertAllCloseAccordingToType(jacob_sym, jacob_num)

    def test_serialization(self, dtype=None):
        ref_fn = sparsemax_loss
        config = tf.keras.losses.serialize(ref_fn)
        fn = tf.keras.losses.deserialize(config)
        self.assertEqual(ref_fn, fn)


if __name__ == "__main__":
    tf.test.main()
