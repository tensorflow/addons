<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.losses.SparsemaxLoss" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
</div>

# tfa.losses.SparsemaxLoss

## Class `SparsemaxLoss`

Sparsemax loss function.





Defined in [`losses/sparsemax_loss.py`](https://github.com/tensorflow/addons/tree/r0.3/tensorflow_addons/losses/sparsemax_loss.py).

<!-- Placeholder for "Used in" -->

Computes the generalized multi-label classification loss for the sparsemax
function.

Because the sparsemax loss function needs both the properbility output and
the logits to compute the loss value, `from_logits` must be `True`.

Because it computes the generalized multi-label loss, the shape of both
`y_pred` and `y_true` must be `[batch_size, num_classes]`.

#### Args:

* <b>`from_logits`</b>: Whether `y_pred` is expected to be a logits tensor. Default
    is `True`, meaning `y_pred` is the logits.
* <b>`reduction`</b>: (Optional) Type of `tf.keras.losses.Reduction` to apply to
    loss. Default value is `SUM_OVER_BATCH_SIZE`.
* <b>`name`</b>: Optional name for the op.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    from_logits=True,
    reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
    name='sparsemax_loss'
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





