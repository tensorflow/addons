<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.losses.sparsemax_loss" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.losses.sparsemax_loss


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.6/tensorflow_addons/losses/sparsemax_loss.py#L26-L78">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Sparsemax loss function [1].

``` python
tfa.losses.sparsemax_loss(
    logits,
    sparsemax,
    labels,
    name=None
)
```



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
