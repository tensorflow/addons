<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.activations.hardshrink" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.activations.hardshrink


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.6/tensorflow_addons/activations/hardshrink.py#L28-L45">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Hard shrink function.

``` python
tfa.activations.hardshrink(
    x,
    lower=-1.0,
    upper=1.0
)
```



<!-- Placeholder for "Used in" -->

Computes hard shrink function:
`x if x < lower or x > upper else 0`.

#### Args:


* <b>`x`</b>: A `Tensor`. Must be one of the following types:
    `float16`, `float32`, `float64`.
* <b>`lower`</b>: `float`, lower bound for setting values to zeros.
* <b>`upper`</b>: `float`, upper bound for setting values to zeros.

#### Returns:

A `Tensor`. Has the same type as `x`.
