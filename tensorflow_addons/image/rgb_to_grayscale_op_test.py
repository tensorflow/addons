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
import sys
from skimage.color import rgb2gray


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def grayscale_test():
    img = tf.random.uniform([1, 40, 40, 3], minval=0, maxval=255, dtype=tf.float64)
    img_n = img.numpy()
    img_n = np.reshape(img_n, [40, 40, 3])
    grayscale_tfa = rgb_to_grayscale(img)
    grayscale_ski = rgb2gray(img_n)
    grayscale_tfa = grayscale_tfa.numpy()
    grayscale_tfa = tf.reshape(grayscale_tfa, [40, 40])
    accuracy = 0
    for i in range(len(grayscale_tfa)):
        for j in range(len(grayscale_tfa[0])):
            if abs(grayscale_tfa[i][j] - grayscale_ski[i][j]) < 20:
                accuracy += 1
    print(accuracy / 1600)
    assert accuracy > 0.85


grayscale_test()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
