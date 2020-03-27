from gaussianBlur_ops import GaussianBlur
import tensorflow as tf



def test(dtype,batch,hight,width):
	test_image=tf.ones([batch,width,hight,1],dtype=dtype)
	gb=GaussianBlur(test_image,1,7)
	print(gb.convolve())
	
test(tf.float64,1,40,40)
	

