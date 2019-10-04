<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.activations.sparsemax" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.activations.sparsemax


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.6/tensorflow_addons/activations/sparsemax.py#L25-L70">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Sparsemax activation function [1].

### Aliases:

* `tfa.layers.sparsemax.sparsemax`


``` python
tfa.activations.sparsemax(
    logits,
    axis=-1
)
```



<!-- Placeholder for "Used in" -->

For each batch `i` and class `j` we have
  $$sparsemax[i, j] = max(logits[i, j] - tau(logits[i, :]), 0)$$

[1]: https://arxiv.org/abs/1602.02068

#### Args:


* <b>`logits`</b>: Input tensor.
* <b>`axis`</b>: Integer, axis along which the sparsemax operation is applied.

#### Returns:

Tensor, output of sparsemax transformation. Has the same type and
shape as `logits`.


#### Raises:


* <b>`ValueError`</b>: In case `dim(logits) == 1`.