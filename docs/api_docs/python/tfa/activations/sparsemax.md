<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.activations.sparsemax" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.activations.sparsemax

Sparsemax activation function [1].

### Aliases:

* `tfa.activations.sparsemax`
* `tfa.layers.sparsemax.sparsemax`

``` python
tfa.activations.sparsemax(
    logits,
    axis=-1,
    name=None
)
```



Defined in [`activations/sparsemax.py`](https://github.com/tensorflow/addons/tree/r0.3/tensorflow_addons/activations/sparsemax.py).

<!-- Placeholder for "Used in" -->

For each batch `i` and class `j` we have
  $$sparsemax[i, j] = max(logits[i, j] - tau(logits[i, :]), 0)$$

[1]: https://arxiv.org/abs/1602.02068

#### Args:

* <b>`logits`</b>: Input tensor.
* <b>`axis`</b>: Integer, axis along which the sparsemax operation is applied.
* <b>`name`</b>: A name for the operation (optional).

#### Returns:

Tensor, output of sparsemax transformation. Has the same type and
shape as `logits`.

#### Raises:

* <b>`ValueError`</b>: In case `dim(logits) == 1`.