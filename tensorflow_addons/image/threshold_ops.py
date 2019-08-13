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
"""Image thresholding ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

@tf.function
def basic_threshold(image, threshold):
	"""Applies a basic threshold operation and outputs a binary image.
	
	Args:
	  image: A tensor of shape (num_rows, num_columns, num_channels) (HWC) or
	  (num_rows, num_columns) (HW). The rank must be statically known (the
	  shape is not `TensorShape(None)`.

	  threshold: An integer value between 0 and 255 which is used to binarize the image by 
	  comparing it with the pixel values.
	
	Returns:
	  Binary thresholded image based on the threshold value 

	Raises:
	  TypeError: If `image` is an invalid type.
	"""
	with tf.name_scope(name or "basic_threshold"):
		image = tf.convert_to_tensor(image, name="image")
		
		rank = image.shape.rank
		if rank != 2 and rank != 3:
			raise ValueError("Image should be either 2 or 3-dimensional.")
 
		if not isinstance(threshold,int):
			raise ValueError("Threshold value must be an integer.")

		if rank == 3:
			image = tf.image.rgb_to_grayscale(image)
			image = tf.image.convert_image_dtype(image, tf.dtypes.uint8)
			image = tf.squeeze(image,2)
		
		r, c = image.shape
		image = image.numpy()
		final = np.zeros((r,c))
		final[image>threshold] = 255
		return final

@tf.function
def adaptive_threshold(image, window):
	"""
	Applies a basic threshold operation but instead of applying on complete
	image it applies on small regions and outputs a binary image(s). Here 
	threshold is calculated from mean of the region where operation is done. 
	
	Args:
	  image: A tensor of shape (num_rows, num_columns, num_channels) (HWC) or
	  (num_rows, num_columns) (HW). The rank must be statically known (the
	  shape is not `TensorShape(None)`.
	
	  window: An integer value used as the size of the local region.
	Returns:
	  Binary thresholded image.

	Raises:
	  TypeError: If `image` is an invalid type.
	"""
	with tf.name_scope(name or "adaptive_threshold"):
		image = tf.convert_to_tensor(image, name="image")
		
		rank = image.shape.rank
		if rank != 2 and rank != 3:
			raise ValueError("Image should be either 2 or 3-dimensional.")
 
		if not isinstance(window,int):
			raise ValueError("Window size value must be an integer.")

		r, c = image.shape
		if window > min(r,c):
			raise ValueError("Window size should be lesser than the size of the image.")

		if rank == 3:
			image = tf.image.rgb_to_grayscale(image)
			image = tf.image.convert_image_dtype(image, tf.dtypes.uint8)
			image = tf.squeeze(image,2)
		image = image.numpy()

		i = 0
		final = np.zeros((r,c))
		while i<r:
			j = 0
			r1 = min(i+window, r)
			while j<c:
				c1 = min(j+window, c)
				cur = image[i:r1,j:c1]
				threshold = np.mean(np.mean(cur,axis=0))
				res = np.zeros_like(cur)
				res[cur>threshold] = 255
				final[i:r1,j:c1] = res

				j += window
			i += window
		return final

@tf.function
def otsu_thresholding(image):
	"""
	Applies thresholding on any image using otsu's method which exhaustively
	searches for the threshold that minimizes the intra-class variance. 
	
	Args:
	  image: A tensor of shape (num_rows, num_columns, num_channels) (HWC) or
	  (num_rows, num_columns) (HW). The rank must be statically known (the
	  shape is not `TensorShape(None)`.	
	Returns:
	  Binary thresholded image.

	Raises:
	  TypeError: If `image` is an invalid type.
	"""
	with tf.name_scope(name or "otsu_thresholding"):
		image = tf.convert_to_tensor(image, name="image")

		rank = image.shape.rank
		if rank != 2 and rank != 3:
			raise ValueError("Image should be either 2 or 3-dimensional.")

		if image.dtype!=tf.int32:
			image = tf.cast(image, tf.int32)

		r, c = image.shape
		hist = tf.math.bincount(image, dtype=tf.int32)

		current_max, threshold = 0, 0
		total = r * c

		hist = hist.numpy()
		spre = [0]*256
		sw = [0]*256
		spre[0] = hist[0]

		for i in range(1,256):
			spre[i] = spre[i-1] + hist[i]
			sw[i] = sw[i-1]  + (i * hist[i])

		for i in range(256):
			if total - spre[i] == 0:
				break

			meanB = sw[i]/spre[i]
			meanF = (sw[255] - sw[i])/(total - spre[i])
			varBetween = (total - spre[i]) * spre[i] * ((meanB-meanF)**2)

			if varBetween > current_max:
				current_max = varBetween
				threshold = i

		final = np.zeros((r,c))
		image = image.numpy()
		final[image>threshold] = 255

		return final