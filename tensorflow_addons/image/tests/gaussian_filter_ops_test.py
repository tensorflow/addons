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
""" Tests for gaussian blur. """


from tensorflow_addons.image.gaussian_filter_ops import gaussian_blur
import numpy as np
import numpy.core.multiarray
from scipy.ndimage import gaussian_filter
import tensorflow as tf
import pytest


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_gaussian_blur():
    test_image_tf = tf.random.uniform(
        [1, 40, 40, 1], minval=0, maxval=255, dtype=tf.float64
    )
    gb = gaussian_blur(test_image_tf, 1, 5)
    gb = gb.numpy()
    gb1 = np.resize(gb, (40, 40))
    test_image_cv = test_image_tf.numpy()
    test_image_cv = np.resize(test_image_cv, [40, 40])
    gb2 = gaussian_filter(test_image_cv, 1, truncate=4.6, mode="constant", cval=0)
    gb1 = gb1[:, :] / 255
    gb2 = gb2[:, :] / 255
    np.testing.assert_allclose(gb2, gb1, 0.5)
