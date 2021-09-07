import unittest
from tensorflow_addons.losses import dice_loss 
import numpy as np

class TestDiceLoss(unittest.TestCase):
	def test_dice_loss(self):
		y_true = np.array([[0,0,1,0],[0,0,1,0],[0,0,1.,0.]])
		y_pred = np.array([[0,0,0.9,0],[0,0,0.1,0],[1,1,0.1,1.]])

		result=dice_loss(y_true, y_pred)
		print("result= \t ",result)
		print("it works")


if __name__ == '__main__':
	unittest.main()


