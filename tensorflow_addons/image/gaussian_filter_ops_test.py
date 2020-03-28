from tensorflow_addons.image.gaussian_filter_ops import gaussian_blur
#from gaussian_filter_ops import gaussian_blur
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import pytest
fp1=open("file1.txt","w")
fp2=open("file2.txt","w")


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
	
test()

