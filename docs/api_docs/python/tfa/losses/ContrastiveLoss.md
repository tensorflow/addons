<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.losses.ContrastiveLoss" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
</div>

# tfa.losses.ContrastiveLoss

## Class `ContrastiveLoss`

Computes the contrastive loss between `y_true` and `y_pred`.



### Aliases:

* Class `tfa.losses.ContrastiveLoss`
* Class `tfa.losses.contrastive.ContrastiveLoss`



Defined in [`losses/contrastive.py`](https://github.com/tensorflow/addons/tree/r0.3/tensorflow_addons/losses/contrastive.py).

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

``` python
__init__(
    margin=1.0,
    reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
    name='contrasitve_loss'
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

* <b>`y_true`</b>: Ground truth values.
* <b>`y_pred`</b>: The predicted values.
* <b>`sample_weight`</b>: Optional `Tensor` whose rank is either 0, or the same rank
    as `y_true`, or is broadcastable to `y_true`. `sample_weight` acts as a
    coefficient for the loss. If a scalar is provided, then the loss is
    simply scaled by the given value. If `sample_weight` is a tensor of size
    `[batch_size]`, then the total loss for each sample of the batch is
    rescaled by the corresponding element in the `sample_weight` vector. If
    the shape of `sample_weight` matches the shape of `y_pred`, then the
    loss of each measurable element of `y_pred` is scaled by the
    corresponding value of `sample_weight`.


#### Returns:

Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
  shape as `y_true`; otherwise, it is scalar.


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

``` python
get_config()
```





