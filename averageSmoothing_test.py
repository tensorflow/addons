from averageSmoothing import AverageSmooth
import tensorflow as tf



def test(dtype,batch,hight,width):
	test_image=tf.ones([batch,width,hight,1],dtype=dtype)
	gb=AverageSmooth(test_image,(3,4))
	print(gb.convolve())
	
test(tf.float64,1,40,40)
	

