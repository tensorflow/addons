<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.losses.TripletSemiHardLoss" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
</div>

# tfa.losses.TripletSemiHardLoss

## Class `TripletSemiHardLoss`

Computes the triplet loss with semi-hard negative mining.



### Aliases:

* Class `tfa.losses.TripletSemiHardLoss`
* Class `tfa.losses.triplet.TripletSemiHardLoss`



Defined in [`losses/triplet.py`](https://github.com/tensorflow/addons/tree/0.4-release/tensorflow_addons/losses/triplet.py).

<!-- Placeholder for "Used in" -->

The loss encourages the positive distances (between a pair of embeddings
with the same labels) to be smaller than the minimum negative distance
among which are at least greater than the positive distance plus the
margin constant (called semi-hard negative) in the mini-batch.
If no such negative exists, uses the largest negative distance instead.
See: https://arxiv.org/abs/1503.03832.

We expect labels `y_true` to be provided as 1-D integer `Tensor` with shape
[batch_size] of multi-class integer labels. And embeddings `y_pred` must be
2-D float `Tensor` of l2 normalized embedding vectors.

#### Args:


* <b>`margin`</b>: Float, margin term in the loss definition. Default value is 1.0.
* <b>`name`</b>: Optional name for the op.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    margin=1.0,
    name=None
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






