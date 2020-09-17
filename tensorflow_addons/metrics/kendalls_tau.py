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
"""Implements Kendall's Tau metric and loss."""

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_addons.metrics.utils import MeanMetricWrapper
from tensorflow_addons.utils.types import TensorLike


def _iterative_mergesort(y: TensorLike, aperm: TensorLike) -> (tf.int32, tf.Tensor):
    """Non-recusive mergesort that counts exchanges.

    Args:
        y: values to be sorted.
        aperm: original ordering.

    Returns:
        A tuple consisting of a int32 scalar that counts the number of
        exchanges required to produce a sorted permutation, and a tf.int32
        Tensor that contains the ordering of y values that are sorted.
    """
    exchanges = 0
    num = tf.size(y)
    k = tf.constant(1, tf.int32)
    while tf.less(k, num):
        for left in tf.range(0, num - k, 2 * k, dtype=tf.int32):
            rght = left + k
            rend = tf.minimum(rght + k, num)
            tmp = tf.TensorArray(dtype=tf.int32, size=num)
            m, i, j = 0, left, rght
            while tf.less(i, rght) and tf.less(j, rend):
                permij = aperm.gather([i, j])
                yij = tf.gather(y, permij)
                if tf.less_equal(yij[0], yij[1]):
                    tmp = tmp.write(m, permij[0])
                    i += 1
                else:
                    tmp = tmp.write(m, permij[1])
                    # Explanation here
                    # https://www.geeksforgeeks.org/counting-inversions/.
                    exchanges += rght - i
                    j += 1
                m += 1
            while tf.less(i, rght):
                tmp = tmp.write(m, aperm.read(i))
                i += 1
                m += 1
            while tf.less(j, rend):
                tmp = tmp.write(m, aperm.read(j))
                j += 1
                m += 1
            aperm = aperm.scatter(tf.range(left, rend), tmp.gather(tf.range(0, m)))
        k *= 2
    return exchanges, aperm.stack()


def kendalls_tau(y_true: TensorLike, y_pred: TensorLike) -> tf.Tensor:
    """Computes Kendall's Tau for two ordered lists.

    Kendall's Tau measures the correlation between ordinal rankings. This
    implementation is similar to the one used in scipy.stats.kendalltau.
    Args:
        y_true: A tensor that provides a true ordinal ranking of N items.
        y_pred: A presumably model provided ordering of the same N items:

    Returns:
        Kendell's Tau, the 1945 tau-b formulation that ignores ordering of
        ties, as a scalar Tensor.
    """
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    y_pred.shape.assert_is_compatible_with(y_true.shape)
    if tf.equal(tf.size(y_true), 0) or tf.equal(tf.size(y_pred), 0):
        return np.nan
    perm = tf.argsort(y_true)
    n = tf.shape(perm)[0]
    if tf.less(n, 2):
        return np.nan

    left = 0
    # scan for ties, and for each range of ties do a argsort on
    # the y_pred value. (TF has no lexicographical sorting, although
    # jax can sort complex number lexicographically. Hmm.)
    lexi = tf.TensorArray(tf.int32, size=n)
    for i in tf.range(n):
        lexi = lexi.write(i, perm[i])
    for right in tf.range(1, n):
        ytruelr = tf.gather(y_true, tf.gather(perm, [left, right]))
        if tf.not_equal(ytruelr[0], ytruelr[1]):
            sub = perm[left:right]
            subperm = tf.argsort(tf.gather(y_pred, sub))
            lexi = lexi.scatter(tf.range(left, right), tf.gather(sub, subperm))
            left = right
    sub = perm[left:n]
    subperm = tf.argsort(tf.gather(y_pred, perm[left:n]))
    lexi.scatter(tf.range(left, n), tf.gather(sub, subperm))

    # This code follows roughly along with scipy/stats/stats.py v. 0.15.1
    # compute joint ties
    first = 0
    t = 0
    for i in tf.range(1, n):
        permfirsti = lexi.gather([first, i])
        y_truefirsti = tf.gather(y_true, permfirsti)
        y_predfirsti = tf.gather(y_pred, permfirsti)
        if y_truefirsti[0] != y_truefirsti[1] or y_predfirsti[0] != y_predfirsti[1]:
            t += ((i - first) * (i - first - 1)) // 2
            first = i
    t += ((n - first) * (n - first - 1)) // 2

    # compute ties in y_true
    first = 0
    u = 0
    for i in tf.range(1, n):
        y_truefirsti = tf.gather(y_true, lexi.gather([first, i]))
        if y_truefirsti[0] != y_truefirsti[1]:
            u += ((i - first) * (i - first - 1)) // 2
            first = i
    u += ((n - first) * (n - first - 1)) // 2

    # count exchanges
    exchanges, newperm = _iterative_mergesort(y_pred, lexi)
    # compute ties in y_pred after mergesort with counting
    first = 0
    v = 0
    for i in tf.range(1, n):
        y_predfirsti = tf.gather(y_pred, tf.gather(newperm, [first, i]))
        if y_predfirsti[0] != y_predfirsti[1]:
            v += ((i - first) * (i - first - 1)) // 2
            first = i
    v += ((n - first) * (n - first - 1)) // 2

    tot = (n * (n - 1)) // 2
    if tf.equal(tot, u) or tf.equal(tot, v):
        return np.nan  # Special case for all ties in both ranks

    # Prevent overflow; equal to np.sqrt((tot - u) * (tot - v))
    denom = tf.math.exp(
        0.5
        * (
            tf.math.log(tf.cast(tot - u, tf.float32))
            + tf.math.log(tf.cast(tot - v, tf.float32))
        )
    )
    tau = (
        tf.cast(tot - (v + u - t), tf.float32) - 2.0 * tf.cast(exchanges, tf.float32)
    ) / denom

    return tau


class KendallsTau(MeanMetricWrapper):
    """Computes how well a model orders items, computing mean-tau for batches.

    Any types supported by tf.math.less may be used for y_pred and y_true
    values, and these types do not need to be the same as they are never
    compared against each other. The return type of this metric is always
    tf.float32 and a value between -1.0 and 1.0.

    References:
    "A Note on Average Tau as a Measure of Concordance", William L Hays,
    Journal of the American Statistical Assoc, Jun 1960, V55 N290 p. 331-341.

    "Statistical Properties of Average Kendall's Tau Under Multivariate
    Contaminated Gaussian Model", Huadong Lai and Weichao Xu, IEEE Access, V7,
    p.159177-159189, 2019.

    Note that there is a streaming implementation of an approximate algorithm,
    but this is different from the one implemented here, see:
    "An Online Algorithm for Nonparametric Correlations", Wei Xiao, 2017,
    https://arxiv.org/abs/1712.01521.

    Attributes:
        name: (Optional) string name of the metric instance.
    """

    def __init__(self, name: str = "kendalls_tau", **kwargs):
        super().__init__(kendalls_tau, name=name, dtype=tf.float32)
