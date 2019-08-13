# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may noa use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow_addons.image import basic_threshold
from tensorflow_addons.image import adaptive_threshold
from tensorflow_addons.image import otsu_thresholding
from tensorflow_addons.utils import test_utils

class BasicThresholdTest(tf.test.TestCase):
	def test_invalid_image(self):
		msg = 'Image should be either 2 or 3-dimensional.'

		for image_shape in [(28, 28), (36,45,3), (16, 28, 28, 1, 1)]:
			with self.subTest(dim=len(image_shape)):
				with self.assertRaisesRegexp(ValueError, msg):
					basic_threshold(tf.ones(shape=image_shape),127)

	def test_thresholding_grayscale(self):
		image = tf.constant([[5, 150, 220, 49, 110],
							[81, 251, 20, 180, 99],
							[239, 7, 129, 47, 235],
							[88, 84, 115, 171, 75],
							[145, 150, 250, 64, 32]])
		threshold = 127

		expected_output = np.array([[0, 255, 255, 0, 0],
									[0, 255, 0, 255, 0],
									[255, 0, 255, 0, 255],
									[0, 0, 0, 255, 0],
									[255, 255, 255, 0, 0]])

		output = basic_threshold(image,threshold)
		self.assertAllClose(output,expected_output)

class AdaptiveThresholdTest(tf.test.TestCase):
	def test_invalid_image(self):
		msg = 'Image should be either 2 or 3-dimensional.'

		for image_shape in [(28, 28), (36,45,3), (16, 28, 28, 1, 1)]:
			with self.subTest(dim=len(image_shape)):
				with self.assertRaisesRegexp(ValueError, msg):
					adaptive_threshold(tf.ones(shape=image_shape),5)

	def test_window_size(self):
		msg = 'Window size should be lesser than the size of the image.'
		image_shape = (25,25)
		for window_size in [3, 5, 15, 30, 10]:
			with self.assertRaisesRegexp(ValueError, msg):
				adaptive_threshold(tf.ones(shape=image_shape),window_size)

class OtsuThresholdTest(tf.test.TestCase):
	def test_invalid_image(self):
		msg = 'Image should be either 2 or 3-dimensional.'

		for image_shape in [(28, 28), (36,45,3), (16, 28, 28, 1, 1)]:
			with self.subTest(dim=len(image_shape)):
				with self.assertRaisesRegexp(ValueError, msg):
					otsu_thresholding(tf.ones(shape=image_shape))
