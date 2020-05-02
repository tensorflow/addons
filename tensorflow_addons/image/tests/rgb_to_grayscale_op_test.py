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
"rgb to grayscale op test"


import numpy as np
import tensorflow as tf
from tensorflow_addons.image.rgb_to_grayscale_op import rgb_to_grayscale
import pytest


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_rgb_to_grayscale_op():
    img = tf.constant(
        [
            [
                [
                    [0.0, 0.0, 0.0],
                    [2.0, 2.0, 2.0],
                    [4.0, 4.0, 4.0],
                    [6.0, 6.0, 6.0],
                    [8.0, 8.0, 8.0],
                ],
                [
                    [10.0, 10.0, 10.0],
                    [12.0, 12.0, 12.0],
                    [14.0, 14.0, 14.0],
                    [16.0, 16.0, 16.0],
                    [18.0, 18.0, 18.0],
                ],
                [
                    [20.0, 20.0, 20.0],
                    [22.0, 22.0, 22.0],
                    [24.0, 24.0, 24.0],
                    [26.0, 26.0, 26.0],
                    [28.0, 28.0, 28.0],
                ],
                [
                    [30.0, 30.0, 30.0],
                    [32.0, 32.0, 32.0],
                    [34.0, 34.0, 34.0],
                    [36.0, 36.0, 36.0],
                    [38.0, 38.0, 38.0],
                ],
                [
                    [40.0, 40.0, 40.0],
                    [42.0, 42.0, 42.0],
                    [44.0, 44.0, 44.0],
                    [46.0, 46.0, 46.0],
                    [48.0, 48.0, 48.0],
                ],
            ]
        ],
        dtype=tf.float32,
    )
    r = np.arange(0, 50, 2)
    count = 0
    exp_plane = np.ones([5, 5, 1])
    for i in range(5):
        for j in range(5):
            exp_plane[i][j][0] = (
                0.2989 * r[count] + 0.5870 * r[count] + 0.1140 * r[count]
            )
            count += 1
    grayscale_tfa = rgb_to_grayscale(img)
    grayscale_tfa = grayscale_tfa.numpy()
    grayscale_tfa = np.resize(grayscale_tfa, (5, 5))
    exp_plane = np.resize(exp_plane, (5, 5))
    np.testing.assert_allclose(grayscale_tfa, exp_plane, 0.06)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_rgb_to_grayscale_op_random_image():
    img = tf.random.uniform([2, 40, 40, 3], minval=0, maxval=255, dtype=tf.float32)
    exp_plane = tf.image.rgb_to_grayscale(img)
    grayscale_tfa = tf.image.rgb_to_grayscale(img)
    np.testing.assert_allclose(grayscale_tfa, exp_plane, 0.06)
