from gaussian_filter_ops import gaussian_blur
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import pytest

def test(dtype,batch,hight,width):
	test_image_tf=tf.zeros([batch,width,hight,1],dtype=dtype)
	gb=gaussian_blur(test_image_tf,1,7)
	gb=gb.numpy()
	gb1=np.resize(gb,(width,hight))
	
	plt.imshow(gb1,'gray')
	#plt.show()
	#print(gb)
	test_image_cv=np.zeros([40,40])
	gb2=cv2.GaussianBlur(test_image_cv,(7,7),1)
	plt.imshow(gb2,'gray')
	#plt.show()
	accuracy=0
	#print(gb1)
	#print(gb2)
	for i in range(len(gb1)):
		for j in range(len(gb1[0])):
			if(gb1[i][j]==gb2[i][j]):
				accuracy+=1
	print("Accuracy w.r.t opencv=",accuracy/(len(gb1)*len(gb1[0])))
	
	
def testWithImage(path):
	img=cv2.imread(path,cv2.IMREAD_GRAYSCALE )
	
	img=np.expand_dims(img,axis=2)
	img=np.expand_dims(img,axis=0)
	img_test_tensor=tf.convert_to_tensor(img,dtype=tf.float64)
	
	gb=gaussian_blur(img_test_tensor,1,7)
	print(img_test_tensor)
	gb=gb.numpy()
	gb1=np.resize(gb,(img.size()[1],img.size()[0]))
	plt.imshow(gb1,'gray')
	plt.show()
test(tf.float64,1,40,30)
#testWithImage("sample_grayscale.jpg")


