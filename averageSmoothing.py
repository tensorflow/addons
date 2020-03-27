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
# =============================================================================
"""AverageSmoothing Op"""


import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import tensor_shape
import tensorflow as tf


def getImageHeightWidth(images,dataFormat):
	if(dataFormat=='channels_last'):
		h,w=tf.shape(images)[1],tf.shape(images)[0]
	else:
		h,w=tf.shape(images)[2], tf.shape(images)[3]
	return h,w

class AverageSmooth():
	"""
	This class is responsible for having Average Smoothing. It takes the image as input, computes an average-smoothing-
	kernel then convolves the image with the kernel.
	Args:
	img: A tensor of shape
        (batch_size, height, width, channels)
        (NHWC), (batch_size, channels, height, width)(NCHW).
    
    kSize:It is the tuple of shape(height,width). 
    		A kernel of size [height*width] is generated.
	"""
	def __init__(self,img,kSize):
		
		
		
		self.kHeight=kSize[0]
		self.kWidth=kSize[1]
		self.gaussianKernelNumpy=(np.ones([self.kHeight,self.kWidth]))
		self.findKernel()
		self.gaussianKernelNumpy=np.expand_dims(self.gaussianKernelNumpy,axis=2)
		self.gaussianKernelNumpy=np.expand_dims(self.gaussianKernelNumpy,axis=2)
		self.img=tf.convert_to_tensor(img)
		self.gaussianKernelTensor=tf.convert_to_tensor(self.gaussianKernelNumpy)
		gaussian_filter_shape=self.gaussianKernelTensor.get_shape()
		self.conv_ops=nn_ops.Convolution(input_shape=img.get_shape(),
										filter_shape=gaussian_filter_shape,
										padding='SAME')
		
		
	def findKernel(self):
		"This function creates a kernel of size [height*width]"
		for i in range (self.kHeight):
			for j in range(self.kWidth):
				self.gaussianKernelNumpy[i][j]=1/(self.kHeight*self.kWidth)
		return

		
		
	def convolve(self):
		"This function is responsible for convolving the given image with the Gaussian Kernel"
		out=self.conv_ops(self.img,self.gaussianKernelTensor)
		return out
		

#with tf.session() as sess:		


		

