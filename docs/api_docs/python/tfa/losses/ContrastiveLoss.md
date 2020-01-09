<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.losses.ContrastiveLoss" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
</div>

# tfa.losses.ContrastiveLoss

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/losses/contrastive.py#L61-L107">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



<!-- Equality marker -->
## Class `ContrastiveLoss`

Computes the contrastive loss between `y_true` and `y_pred`.



**Aliases**: `tfa.losses.contrastive.ContrastiveLoss`

<!-- Placeholder for "Used in" -->

This loss encourages the embedding to be close to each other for
the samples of the same label and the embedding to be far apart at least
by the margin constant for the samples of different labels.

See: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

We expect labels `y_true` to be provided as 1-D integer `Tensor`
with shape [batch_size] of binary integer labels. And `y_pred` must be
1-D float `Tensor` with shape [batch_size] of distances between two
embedding matrices.

The euclidean distances `y_pred` between two embedding matrices
`a` and `b` with shape [batch_size, hidden_size] can be computed
as follows:

```python
# y_pred = \sqrt (\sum_i (a[:, i] - b[:, i])^2)
y_pred = tf.linalg.norm(a - b, axis=1)
```

#### Args:


* <b>`margin`</b>: `Float`, margin term in the loss definition.
  Default value is 1.0.
* <b>`reduction`</b>: (Optional) Type of `tf.keras.losses.Reduction` to apply.
  Default value is `SUM_OVER_BATCH_SIZE`.
* <b>`name`</b>: (Optional) name for the loss.

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/losses/contrastive.py#L92-L97">View source</a>

``` python
__init__(
    margin=1.0,
    reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
    name='contrasitve_loss'
)
```

Initialize self.  See help(type(self)) for accurate signature.




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

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/losses/contrastive.py#L102-L107">View source</a>

``` python
get_config()
```








