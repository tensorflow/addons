# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Approximate Kendall's Tau-b Metric."""

import tensorflow as tf
from tensorflow.keras.metrics import Metric
from tensorflow_addons.utils.types import AcceptableDTypes

from typeguard import typechecked


@tf.keras.utils.register_keras_serializable(package="Addons")
class KendallsTau(Metric):
    """Computes Kendall's Tau-b Rank Correlation Coefficient.

    A measure of ordinal similarity between equal length sequences
    of values, with allowances for ties.

    Based on the algorithm of Wei Xiao https://arxiv.org/abs/1712.01521.

    Usage:

    ```python
    actuals = tf.constant([12, 2, 1, 12, 2], dtype=np.int32)
    preds = tf.constant([1, 4, 7, 1, 0], dtype=np.int32)

    m = tfa.metrics.KendallsTau(0, 13)
    m.update_state(actuals, preds)
    print('Final result: ', m.result().numpy()) # Result: -0.4714045
    ```

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
        name: str = "kendalls_tau",
        dtype: AcceptableDTypes = None,
    ):
        """Creates a `KendallsTau` instance.

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
        self.reset_states()

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates ranks.

        Args:
          y_true: actual rank values.
          y_pred: predicted rank values.
          sample_weight (optional): Ignored.

        Returns:
          Update op.
        """
        if y_true.shape and y_true.shape[0]:
            i = tf.searchsorted(
                self.actual_cuts,
                tf.cast(tf.reshape(y_true, -1), self.actual_cuts.dtype),
            )
            j = tf.searchsorted(
                self.preds_cuts, tf.cast(tf.reshape(y_pred, -1), self.preds_cuts.dtype)
            )

            def body(k, n, m, nrow, ncol):
                return (
                    k + 1,
                    n + 1,
                    tf.sparse.add(
                        m,
                        tf.SparseTensor(
                            [[i[k], j[k]]],
                            tf.cast([1], dtype=self.m.dtype),
                            self.m.shape,
                        ),
                    ),
                    tf.sparse.add(
                        nrow,
                        tf.SparseTensor(
                            [[i[k]]],
                            tf.cast([1], dtype=self.nrow.dtype),
                            self.nrow.shape,
                        ),
                    ),
                    tf.sparse.add(
                        ncol,
                        tf.SparseTensor(
                            [[j[k]]],
                            tf.cast([1], dtype=self.ncol.dtype),
                            self.ncol.shape,
                        ),
                    ),
                )

            _, self.n, self.m, self.nrow, self.ncol = tf.while_loop(
                lambda k, n, m, nrow, ncol: k < i.shape[0],
                body=body,
                loop_vars=(0, self.n, self.m, self.nrow, self.ncol),
            )

    def result(self):
        m_dense = tf.sparse.to_dense(tf.cast(self.m, tf.float32))
        n_cap = tf.cumsum(
            tf.cumsum(
                tf.slice(tf.pad(m_dense, [[1, 0], [1, 0]]), [0, 0], self.m.shape),
                axis=0,
            ),
            axis=1,
        )
        # Number of concordant pairs.
        p = tf.math.reduce_sum(tf.multiply(n_cap, m_dense))
        sum_m_squard = tf.math.reduce_sum(tf.math.square(m_dense))
        # Ties in x.
        t = (
            tf.math.reduce_sum(tf.math.square(tf.sparse.to_dense(self.nrow)))
            - sum_m_squard
        ) / 2.0
        # Ties in y.
        u = (
            tf.math.reduce_sum(tf.math.square(tf.sparse.to_dense(self.ncol)))
            - sum_m_squard
        ) / 2.0
        # Ties in both.
        b = tf.math.reduce_sum(tf.multiply(m_dense, (m_dense - 1.0))) / 2.0
        # Number of discordant pairs.
        n = tf.cast(self.n, tf.float32)
        q = (n - 1.0) * n / 2.0 - p - t - u - b
        return (p - q) / tf.math.sqrt((p + q + t) * (p + q + u))

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

    def reset_states(self):
        """Resets all of the metric state variables."""
        self.actual_cuts = tf.linspace(
            tf.cast(self.actual_min, tf.float32),
            tf.cast(self.actual_max, tf.float32),
            self.actual_cutpoints - 1,
        )
        self.preds_cuts = tf.linspace(
            tf.cast(self.preds_min, tf.float32),
            tf.cast(self.preds_max, tf.float32),
            self.preds_cutpoints - 1,
        )
        self.m = tf.SparseTensor(
            tf.zeros((0, 2), tf.int64),
            [],
            [self.actual_cutpoints, self.preds_cutpoints],
        )
        self.nrow = tf.SparseTensor(
            tf.zeros((0, 1), dtype=tf.int64), [], [self.actual_cutpoints]
        )
        self.ncol = tf.SparseTensor(
            tf.zeros((0, 1), dtype=tf.int64), [], [self.preds_cutpoints]
        )
        self.n = 0
