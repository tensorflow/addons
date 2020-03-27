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
"""GaussuanBlur Op"""


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

class GaussianBlur():
	"""
	This class is responsible for having Gaussian Blur. It takes the image as input, computes a gaussian-kernel
	which follows normal distribution then convolves the image with the kernel.
	Args:
	img: A tensor of shape
        (batch_size, height, width, channels)
        (NHWC), (batch_size, channels, height, width)(NCHW).
    sigma:A constant of type float64. It is the standard deviation of the normal distribution.
    		The more the sigma, the more the blurring effect.
    		G(x,y)=1/(2*3.14*sigma**2)e^((x**2+y**2)/2sigma**2)
    kSize:It is the kernel-size for the Gaussian Kernel. kSize should be odd.
    		A kernel of size [kSize*kSize] is generated.
	"""
	def __init__(self,img,sigma,kSize):
		if(sigma==0):
			raise ValueError("Sigma should not be zero")
		self.sigma=sigma
		if(kSize%2==0):
			raise ValueError("kSize should be odd")
		
		self.kSize=kSize
		self.gaussianKernelNumpy=(np.zeros([kSize,kSize]))
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
		"This function creates a kernel of size [kSize*kSize]"
		for i in range(-self.kSize//2,self.kSize//2+1):
			for j in range(-self.kSize//2,self.kSize//2+1):
				self.gaussianKernelNumpy[i+self.kSize//2][j+self.kSize//2]=1/(2*np.pi*(self.sigma)**2)*np.exp(-(i**2+j**2)/(2*self.sigma**2))
		return

		
		
	def convolve(self):
		"This function is responsible for convolving the given image with the Gaussian Kernel"
		out=self.conv_ops(self.img,self.gaussianKernelTensor)
		return out
		

#with tf.session() as sess:		


		

