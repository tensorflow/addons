<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.losses.SigmoidFocalCrossEntropy" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
</div>

# tfa.losses.SigmoidFocalCrossEntropy


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.6/tensorflow_addons/losses/focal_loss.py#L27-L98">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



## Class `SigmoidFocalCrossEntropy`

Implements the focal loss function.



### Aliases:

* Class `tfa.losses.focal_loss.SigmoidFocalCrossEntropy`


<!-- Placeholder for "Used in" -->

Focal loss was first introduced in the RetinaNet paper
(https://arxiv.org/pdf/1708.02002.pdf). Focal loss is extremely useful for
classification when you have highly imbalanced classes. It down-weights
well-classified examples and focuses on hard examples. The loss value is
much high for a sample which is misclassified by the classifier as compared
to the loss value corresponding to a well-classified example. One of the
best use-cases of focal loss is its usage in object detection where the
imbalance between the background class and other classes is extremely high.

#### Usage:



```python
fl = tfa.losses.SigmoidFocalCrossEntropy()
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

#### Returns:

Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
    shape as `y_true`; otherwise, it is scalar.



#### Raises:


* <b>`ValueError`</b>: If the shape of `sample_weight` is invalid or value of
  `gamma` is less than zero

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.6/tensorflow_addons/losses/focal_loss.py#L70-L81">View source</a>

``` python
__init__(
    from_logits=False,
    alpha=0.25,
    gamma=2.0,
    reduction=tf.keras.losses.Reduction.NONE,
    name='sigmoid_focal_crossentropy'
)
```






## Methods

<h3 id="__call__"><code>__call__</code></h3>

``` python
__call__(
    y_true,
    y_pred,
    sample_weight=None
)
```

Invokes the `Loss` instance.


#### Args:


* <b>`y_true`</b>: Ground truth values. shape = `[batch_size, d0, .. dN]`
* <b>`y_pred`</b>: The predicted values. shape = `[batch_size, d0, .. dN]`
* <b>`sample_weight`</b>: Optional `sample_weight` acts as a
  coefficient for the loss. If a scalar is provided, then the loss is
  simply scaled by the given value. If `sample_weight` is a tensor of size
  `[batch_size]`, then the total loss for each sample of the batch is
  rescaled by the corresponding element in the `sample_weight` vector. If
  the shape of `sample_weight` is `[batch_size, d0, .. dN-1]` (or can be
  broadcasted to this shape), then each loss element of `y_pred` is scaled
  by the corresponding value of `sample_weight`. (Note on`dN-1`: all loss
  functions reduce by 1 dimension, usually axis=-1.)


#### Returns:

Weighted loss float `Tensor`. If `reduction` is `NONE`, this has
  shape `[batch_size, d0, .. dN-1]`; otherwise, it is scalar. (Note `dN-1`
  because all loss functions reduce by 1 dimension, usually axis=-1.)



#### Raises:


* <b>`ValueError`</b>: If the shape of `sample_weight` is invalid.

<h3 id="from_config"><code>from_config</code></h3>

``` python
from_config(
    cls,
    config
)
```

Instantiates a `Loss` from its config (output of `get_config()`).


#### Args:


* <b>`config`</b>: Output of `get_config()`.


#### Returns:

A `Loss` instance.


<h3 id="get_config"><code>get_config</code></h3>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.6/tensorflow_addons/losses/focal_loss.py#L91-L98">View source</a>

``` python
get_config()
```






