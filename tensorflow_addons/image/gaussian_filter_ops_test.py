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
"""Tests for gaussian blur."""




from tensorflow_addons.image.gaussian_filter_ops import gaussian_blur
#from gaussian_filter_ops import gaussian_blur
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import pytest



def test():
	test_image_tf=tf.random.uniform([1,40,40,1],minval=0,maxval=255,dtype=tf.float64)
	
	gb=gaussian_blur(test_image_tf,1,7)
	gb=gb.numpy()
	gb1=np.resize(gb,(40,40))
	
	
	test_image_cv=test_image_tf.numpy()
	test_image_cv=np.resize(test_image_cv,[40,40])
	
	gb2=cv2.GaussianBlur(test_image_cv,(7,7),1)
	
	accuracy=0
	
	for i in range(len(gb1)):
		for j in range(len(gb1[0])):
			if(abs(gb1[i][j]-gb2[i][j])<10):
				accuracy+=1
	

	print(accuracy/1600)
	assert accuracy>=.80
	
if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
