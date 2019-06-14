<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.losses.sparsemax_loss" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.losses.sparsemax_loss

Sparsemax loss function [1].

``` python
tfa.losses.sparsemax_loss(
    logits,
    sparsemax,
    labels,
    name=None
)
```



Defined in [`losses/sparsemax_loss.py`](https://github.com/tensorflow/addons/tree/0.4-release/tensorflow_addons/losses/sparsemax_loss.py).

<!-- Placeholder for "Used in" -->

Computes the generalized multi-label classification loss for the sparsemax
function. The implementation is a reformulation of the original loss
function such that it uses the sparsemax properbility output instead of the
internal    au variable. However, the output is identical to the original
loss function.

[1]: https://arxiv.org/abs/1602.02068

#### Args:


* <b>`logits`</b>: A `Tensor`. Must be one of the following types: `float32`,
  `float64`.
* <b>`sparsemax`</b>: A `Tensor`. Must have the same type as `logits`.
* <b>`labels`</b>: A `Tensor`. Must have the same type as `logits`.
* <b>`name`</b>: A name for the operation (optional).

#### Returns:

A `Tensor`. Has the same type as `logits`.
