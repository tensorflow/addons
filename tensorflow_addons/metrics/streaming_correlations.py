# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Approximate Pearson's, Spearman's, Kendall's Tau-b/c correlations based
on the algorithm of Wei Xiao https://arxiv.org/abs/1712.01521."""
from abc import abstractmethod

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.metrics import Metric
from tensorflow_addons.utils.types import AcceptableDTypes
from typeguard import typechecked


class CorrelationBase(Metric):
    """Base class for streaming correlation metrics.

    Based on https://arxiv.org/abs/1712.01521.

    It stores and updates the joint and marginal histograms of (`y_true`, `y_pred`).

    The concrete classes estimate the different correlation metrics
    based on those histograms.
    """

    @typechecked
    def __init__(
        self,
        actual_min: float = 0.0,
        actual_max: float = 1.0,
        preds_min: float = 0.0,
        preds_max: float = 1.0,
        actual_cutpoints: int = 100,
        preds_cutpoints: int = 100,
        name: str = None,
        dtype: AcceptableDTypes = None,
    ):
        """Creates a `CorrelationBase` instance.

        Args:
          actual_min: the inclusive lower bound on values from actual.
          actual_max: the exclusive upper bound on values from actual.
          preds_min: the inclusive lower bound on values from preds.
          preds_max: the exclusive upper bound on values from preds.
          actual_cutpoints: the number of divisions to create in actual range,
            defaults to 100.
          preds_cutpoints: the number of divisions to create in preds range,
            defaults to 100.
          name: (optional) String name of the metric instance
          dtype: (optional) Data type of the metric result. Defaults to `None`
        """
        super().__init__(name=name, dtype=dtype)
        self.actual_min = actual_min
        self.actual_max = actual_max
        self.preds_min = preds_min
        self.preds_max = preds_max
        self.actual_cutpoints = actual_cutpoints
        self.preds_cutpoints = preds_cutpoints
        actual_cuts = np.linspace(
            tf.cast(self.actual_min, tf.float32),
            tf.cast(self.actual_max, tf.float32),
            self.actual_cutpoints,
        )
        actual_cuts[-1] += backend.epsilon()
        preds_cuts = np.linspace(
            tf.cast(self.preds_min, tf.float32),
            tf.cast(self.preds_max, tf.float32),
            self.preds_cutpoints,
        )
        preds_cuts[-1] += backend.epsilon()
        self.actual_cuts = tf.convert_to_tensor(actual_cuts, tf.float32)
        self.preds_cuts = tf.convert_to_tensor(preds_cuts, tf.float32)
        self.m = self.add_weight(
            "m", (self.actual_cutpoints - 1, self.preds_cutpoints - 1), dtype=tf.int64
        )
        self.nrow = self.add_weight("nrow", (self.actual_cutpoints - 1), dtype=tf.int64)
        self.ncol = self.add_weight("ncol", (self.preds_cutpoints - 1), dtype=tf.int64)
        self.n = self.add_weight("n", (), dtype=tf.int64)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Updates `m`, `nrow`, `ncol` respectively the joint and
        marginal histograms of (`y_true`, `y_pred`)
        """

        y_true = tf.clip_by_value(y_true, self.actual_min, self.actual_max)
        y_pred = tf.clip_by_value(y_pred, self.preds_min, self.preds_max)

        i = (
            tf.searchsorted(
                self.actual_cuts,
                tf.cast(tf.reshape(y_true, [-1]), self.actual_cuts.dtype),
                side="right",
                out_type=tf.int64,
            )
            - 1
        )
        j = (
            tf.searchsorted(
                self.preds_cuts,
                tf.cast(tf.reshape(y_pred, [-1]), self.preds_cuts.dtype),
                side="right",
                out_type=tf.int64,
            )
            - 1
        )

        nrow = tf.tensor_scatter_nd_add(
            self.nrow, tf.expand_dims(i, axis=-1), tf.ones_like(i)
        )
        ncol = tf.tensor_scatter_nd_add(
            self.ncol, tf.expand_dims(j, axis=-1), tf.ones_like(j)
        )
        ij = tf.stack([i, j], axis=1)
        m = tf.tensor_scatter_nd_add(self.m, ij, tf.ones_like(i))

        self.n.assign_add(tf.shape(i, out_type=tf.int64)[0])
        self.m.assign(m)
        self.nrow.assign(nrow)
        self.ncol.assign(ncol)

    @abstractmethod
    def result(self):
        pass

    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {
            "actual_min": self.actual_min,
            "actual_max": self.actual_max,
            "preds_min": self.preds_min,
            "preds_max": self.preds_max,
            "actual_cutpoints": self.actual_cutpoints,
            "preds_cutpoints": self.preds_cutpoints,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def reset_state(self):
        """Resets all of the metric state variables."""

        self.m.assign(
            tf.zeros((self.actual_cutpoints - 1, self.preds_cutpoints - 1), tf.int64)
        )
        self.nrow.assign(tf.zeros((self.actual_cutpoints - 1), tf.int64))
        self.ncol.assign(tf.zeros((self.preds_cutpoints - 1), tf.int64))
        self.n.assign(0)

    def reset_states(self):
        # Backwards compatibility alias of `reset_state`. New classes should
        # only implement `reset_state`.
        # Required in Tensorflow < 2.5.0
        return self.reset_state()


class KendallsTauBase(CorrelationBase):
    """Base class for kendall's tau metrics."""

    def _compute_variables(self):
        """Compute a tuple containing the concordant pairs, discordant pairs,
        ties in `y_true` and `y_pred`.

        Returns:
          A tuple
        """
        m = tf.cast(self.m, tf.float32)
        n_cap = tf.cumsum(tf.cumsum(m, axis=0), axis=1)
        # Number of concordant pairs.
        p = tf.math.reduce_sum(tf.multiply(n_cap[:-1, :-1], m[1:, 1:]))
        sum_m_squard = tf.math.reduce_sum(tf.math.square(m))
        # Ties in x.
        t = (
            tf.cast(tf.math.reduce_sum(tf.math.square(self.nrow)), tf.float32)
            - sum_m_squard
        ) / 2.0
        # Ties in y.
        u = (
            tf.cast(tf.math.reduce_sum(tf.math.square(self.ncol)), tf.float32)
            - sum_m_squard
        ) / 2.0
        # Ties in both.
        b = tf.math.reduce_sum(tf.multiply(m, (m - 1.0))) / 2.0
        # Number of discordant pairs.
        n = tf.cast(self.n, tf.float32)
        q = (n - 1.0) * n / 2.0 - p - t - u - b
        return p, q, t, u


@tf.keras.utils.register_keras_serializable(package="Addons")
class KendallsTauB(KendallsTauBase):
    """Computes Kendall's Tau-b Rank Correlation Coefficient.

    Usage:
    >>> actuals = tf.constant([12, 2, 1, 12, 2], dtype=tf.int32)
    >>> preds = tf.constant([1, 4, 7, 1, 0], dtype=tf.int32)
    >>> m = tfa.metrics.KendallsTauB(0, 13, 0, 8)
    >>> m.update_state(actuals, preds)
    >>> m.result().numpy()
    -0.47140455
    """

    def result(self):
        p, q, t, u = self._compute_variables()
        return (p - q) / tf.math.sqrt((p + q + t) * (p + q + u))


@tf.keras.utils.register_keras_serializable(package="Addons")
class KendallsTauC(KendallsTauBase):
    """Computes Kendall's Tau-c Rank Correlation Coefficient.

    Usage:
    >>> actuals = tf.constant([12, 2, 1, 12, 2], dtype=tf.int32)
    >>> preds = tf.constant([1, 4, 7, 1, 0], dtype=tf.int32)
    >>> m = tfa.metrics.KendallsTauC(0, 13, 0, 8)
    >>> m.update_state(actuals, preds)
    >>> m.result().numpy()
    -0.48000002
    """

    def result(self):
        p, q, _, _ = self._compute_variables()
        n = tf.cast(self.n, tf.float32)
        non_zeros_col = tf.math.count_nonzero(self.ncol)
        non_zeros_row = tf.math.count_nonzero(self.nrow)
        m = tf.cast(tf.minimum(non_zeros_col, non_zeros_row), tf.float32)
        return 2 * (p - q) / (tf.square(n) * (m - 1) / m)


@tf.keras.utils.register_keras_serializable(package="Addons")
class SpearmansRank(CorrelationBase):
    """Computes Spearman's Rank Correlation Coefficient.

    Usage:
    >>> actuals = tf.constant([12, 2, 1, 12, 2], dtype=tf.int32)
    >>> preds = tf.constant([1, 4, 7, 1, 0], dtype=tf.int32)
    >>> m = tfa.metrics.SpearmansRank(0, 13, 0, 8)
    >>> m.update_state(actuals, preds)
    >>> m.result().numpy()
    -0.54073805
    """

    def result(self):
        nrow = tf.cast(self.nrow, tf.float32)
        ncol = tf.cast(self.ncol, tf.float32)
        n = tf.cast(self.n, tf.float32)

        nrow_ = tf.where(nrow > 0, nrow, -1.0)
        rrow = tf.cumsum(nrow, exclusive=True) + (nrow_ - n) / 2
        ncol_ = tf.where(ncol > 0, ncol, -1.0)
        rcol = tf.cumsum(ncol, exclusive=True) + (ncol_ - n) / 2

        rrow = rrow / tf.math.sqrt(tf.reduce_sum(nrow * tf.square(rrow)))
        rcol = rcol / tf.math.sqrt(tf.reduce_sum(ncol * tf.square(rcol)))

        m = tf.cast(self.m, tf.float32)
        corr = tf.matmul(tf.expand_dims(rrow, axis=0), m)
        corr = tf.matmul(corr, tf.expand_dims(rcol, axis=1))
        return tf.squeeze(corr)


@tf.keras.utils.register_keras_serializable(package="Addons")
class PearsonsCorrelation(CorrelationBase):
    """Computes Pearsons's Correlation Coefficient.

    Usage:
    >>> actuals = tf.constant([12, 2, 1, 12, 2], dtype=tf.int32)
    >>> preds = tf.constant([1, 4, 7, 1, 0], dtype=tf.int32)
    >>> m = tfa.metrics.PearsonsCorrelation(0, 13, 0, 8)
    >>> m.update_state(actuals, preds)
    >>> m.result().numpy()
    -0.5618297
    """

    def result(self):
        ncol = tf.cast(self.ncol, tf.float32)
        nrow = tf.cast(self.nrow, tf.float32)
        m = tf.cast(self.m, tf.float32)
        n = tf.cast(self.n, tf.float32)

        col_bins = (self.preds_cuts[1:] - self.preds_cuts[:-1]) / 2.0 + self.preds_cuts[
            :-1
        ]
        row_bins = (
            self.actual_cuts[1:] - self.actual_cuts[:-1]
        ) / 2.0 + self.actual_cuts[:-1]

        n_col = tf.reduce_sum(ncol)
        n_row = tf.reduce_sum(nrow)
        col_mean = tf.reduce_sum(ncol * col_bins) / n_col
        row_mean = tf.reduce_sum(nrow * row_bins) / n_row

        col_var = tf.reduce_sum(ncol * tf.square(col_bins)) - n_col * tf.square(
            col_mean
        )
        row_var = tf.reduce_sum(nrow * tf.square(row_bins)) - n_row * tf.square(
            row_mean
        )

        joint_product = m * tf.expand_dims(row_bins, axis=1) * col_bins

        corr = tf.reduce_sum(joint_product) - n * col_mean * row_mean

        return corr / tf.sqrt(col_var * row_var)
