<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.activations.softshrink" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.activations.softshrink

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/activations/softshrink.py#L27-L43">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



<!-- Equality marker -->
Soft shrink function.

``` python
tfa.activations.softshrink(
    x,
    lower=-0.5,
    upper=0.5
)
```



<!-- Placeholder for "Used in" -->

Computes soft shrink function:
`x - lower if x < lower, x - upper if x > upper else 0`.

#### Args:


* <b>`x`</b>: A `Tensor`. Must be one of the following types:
    `float16`, `float32`, `float64`.
* <b>`lower`</b>: `float`, lower bound for setting values to zeros.
* <b>`upper`</b>: `float`, upper bound for setting values to zeros.

#### Returns:

A `Tensor`. Has the same type as `x`.


