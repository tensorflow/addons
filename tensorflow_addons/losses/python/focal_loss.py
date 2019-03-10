# -*- coding: utf-8 -*-

import abc

import six
import math
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.keras.utils.losses_utils import compute_weighted_loss
from tensorflow.python.keras.utils.tf_utils import is_tensor_or_variable
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls
from tensorflow.python.keras.utils import losses_utils

import tensorflow as tf
import tensorflow.keras.backend as K


@keras_export('keras.losses.Loss')
class Loss(object):
  """Loss base class.
  To be implemented by subclasses:
  * `call()`: Contains the logic for loss calculation using `y_true`, `y_pred`.
  Example subclass implementation:
  ```
  class MeanSquaredError(Loss):
    def call(self, y_true, y_pred):
      y_pred = ops.convert_to_tensor(y_pred)
      y_true = math_ops.cast(y_true, y_pred.dtype)
      return K.mean(math_ops.square(y_pred - y_true), axis=-1)
  ```
  Args:
    reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to loss.
      Default value is `SUM_OVER_BATCH_SIZE`.
    name: Optional name for the op.
  """

  def __init__(self,
               reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
               name=None):
    self.reduction = reduction
    self.name = name

  def __call__(self, y_true, y_pred, sample_weight=None):
    """Invokes the `Loss` instance.
    Args:
      y_true: Ground truth values.
      y_pred: The predicted values.
      sample_weight: Optional `Tensor` whose rank is either 0, or the same rank
        as `y_true`, or is broadcastable to `y_true`. `sample_weight` acts as a
        coefficient for the loss. If a scalar is provided, then the loss is
        simply scaled by the given value. If `sample_weight` is a tensor of size
        `[batch_size]`, then the total loss for each sample of the batch is
        rescaled by the corresponding element in the `sample_weight` vector. If
        the shape of `sample_weight` matches the shape of `y_pred`, then the
        loss of each measurable element of `y_pred` is scaled by the
        corresponding value of `sample_weight`.
    Returns:
      Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
        shape as `y_true`; otherwise, it is scalar.
    Raises:
      ValueError: If the shape of `sample_weight` is invalid.
    """
    # If we are wrapping a lambda function strip '<>' from the name as it is not
    # accepted in scope name.
    scope_name = 'lambda' if self.name == '<lambda>' else self.name
    with ops.name_scope(scope_name, format(self.__class__.__name__),
                        (y_pred, y_true, sample_weight)):
      losses = self.call(y_true, y_pred)
      return losses_utils.compute_weighted_loss(
          losses, sample_weight, reduction=self.reduction)

  @classmethod
  def from_config(cls, config):
    """Instantiates a `Loss` from its config (output of `get_config()`).
    Args:
        config: Output of `get_config()`.
    Returns:
        A `Loss` instance.
    """
    return cls(**config)

  def get_config(self):
    return {'reduction': self.reduction, 'name': self.name}

  @abc.abstractmethod
  @doc_controls.for_subclass_implementers
  def call(self, y_true, y_pred):
    """Invokes the `Loss` instance.
    Args:
      y_true: Ground truth values, with the same shape as 'y_pred'.
      y_pred: The predicted values.
    """
    NotImplementedError('Must be implemented in subclasses.')


class LossFunctionWrapper(Loss):
  """Wraps a loss function in the `Loss` class.
  Args:
    fn: The loss function to wrap, with signature `fn(y_true, y_pred,
      **kwargs)`.
    reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to loss.
      Default value is `SUM_OVER_BATCH_SIZE`.
    name: (Optional) name for the loss.
    **kwargs: The keyword arguments that are passed on to `fn`.
  """

  def __init__(self,
               fn,
               reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
               name=None,
               **kwargs):
    super(LossFunctionWrapper, self).__init__(reduction=reduction, name=name)
    self.fn = fn
    self._fn_kwargs = kwargs

  def call(self, y_true, y_pred):
    """Invokes the `LossFunctionWrapper` instance.
    Args:
      y_true: Ground truth values.
      y_pred: The predicted values.
    Returns:
      Loss values per sample.
    """
    return self.fn(y_true, y_pred, **self._fn_kwargs)

  def get_config(self):
    config = {}
    for k, v in six.iteritems(self._fn_kwargs):
      config[k] = K.eval(v) if is_tensor_or_variable(v) else v
    base_config = super(LossFunctionWrapper, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))





@keras_export('keras.losses.SigmoidFocalCrossEntropy')
class SigmoidFocalCrossEntropy(LossFunctionWrapper):
  """Focal loss down-weights well classified examples and focuses on the hard
  examples. See https://arxiv.org/pdf/1708.02002.pdf for the loss definition.

  Usage:

  ```python
  fl = tf.keras.losses.SigmoidFocalCrossEntropy()
  loss = fl(
    [[0.97], [0.91], [0.03]],
    [[1], [1], [0])
  print('Loss: ', loss.numpy())  # Loss: [[0.03045921]
                                          [0.09431068]
                                          [0.31471074]
  ```
  Usage with tf.keras API:

  ```python
  model = tf.keras.Model(inputs, outputs)
  model.compile('sgd', loss=tf.keras.losses.SigmoidFocalCrossEntropy())
  ```

  Args
    alpha: balancing factor, default value is 0.25
    gamma: modulating factor, default value is 2.0
  
  Returns:
    Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
        shape as `y_true`; otherwise, it is scalar.
  
  Raises:
      ValueError: If the shape of `sample_weight` is invalid or value of 
        `gamma` is less than zero


  """

  def __init__(self,
               from_logits=False,
               alpha=0.25,
               gamma=2.0,
               reduction=losses_utils.ReductionV2.NONE,
               name='sigmoid_focal_crossentropy'):
    super(SigmoidFocalCrossEntropy, self).__init__(
        sigmoid_focal_crossentropy,
        name=name,
        reduction=reduction,
        from_logits=from_logits,
        alpha=alpha,
        gamma=gamma)

    self.from_logits = from_logits
    self.alpha = alpha
    self.gamma = gamma

@keras_export('keras.metrics.sigmoid_focal_crossentropy',
              'keras.losses.sigmoid_focal_crossentropy',
              'keras.losses.focal_loss')
def sigmoid_focal_crossentropy(y_true, 
                                y_pred, 
                                alpha=0.25,
                                gamma=2.0,
                                from_logits=False):
    """
    Args
        y_true: true targets tensor.
        y_pred: predictions tensor.
        alpha: balancing factor.
        gamma: modulating factor.
    
    Returns:
        Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
        shape as `y_true`; otherwise, it is scalar.
    """
    if gamma:
        if gamma < 0.: 
            raise ValueError("Value of gamma should be greater than or equal to zero")

    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)

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
        alpha = ops.convert_to_tensor(alpha, dtype=K.floatx())
        alpha_factor = y_true*alpha + ((1-alpha)*(1-y_true))
    
    if gamma:
        gamma = ops.convert_to_tensor(gamma, dtype=K.floatx())
        modulating_factor = K.pow((1-p_t), gamma)

    # compute the final loss and return
    return K.mean(alpha_factor*modulating_factor*bce, axis=-1)
