from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import backend as K


def sigmoid_focal_crossentropy(y_true, y_pred, alpha=0.25,
                               gamma=2.0,
                               from_logits=False):
  """Focal loss down-weights well classified examples and focuses on the hard
  examples. See https://arxiv.org/pdf/1708.02002.pdf for the loss definition.

  # Args
      y_true: tensor of true targets.
      y_pred: tensor of predicted targets.
      alpha: balancing factor.
      gamma: modulating factor.

  # Returns:
      loss: a float tensor of shape [batch_size, num_anchors, num_classes]
              representing the value of the loss function.
  """
  assert gamma >= 0, "Value of gamma should be greater than or equal to zero"

  def focal_loss(y_true, y_pred, from_logits=False):
    # Get the binary cross_entropy
    bce = K.binary_crossentropy(y_true, y_pred, from_logits=from_logits)

    # If logits are provided, compute convert the predictions into probabilities
    if from_logits:
      y_pred = K.sigmoid(y_pred)
    else:
      y_pred = K.clip(y_pred, K.epsilon(), 1.- K.epsilon())

    p_t = (y_true*y_pred) + ((1-y_true)*(1-y_pred))
    alpha_factor = 1
    modulating_factor = 1

    if alpha:
      alpha_factor = y_true*alpha + ((1-alpha)*(1-y_true))
    if gamma:
      modulating_factor = K.pow((1-p_t), gamma)

    # compute the final loss and return
    return K.mean(alpha_factor*modulating_factor*bce)
  return focal_loss(y_true, y_pred, from_logits)
