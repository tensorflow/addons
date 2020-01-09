<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.activations.lisht" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.activations.lisht

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/activations/lisht.py#L27-L42">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



<!-- Equality marker -->
LiSHT: Non-Parameteric Linearly Scaled Hyperbolic Tangent Activation Function.

``` python
tfa.activations.lisht(x)
```



<!-- Placeholder for "Used in" -->

Computes linearly scaled hyperbolic tangent (LiSHT): `x * tanh(x)`

See [LiSHT: Non-Parameteric Linearly Scaled Hyperbolic Tangent Activation Function for Neural Networks](https://arxiv.org/abs/1901.05894).

#### Args:


* <b>`x`</b>: A `Tensor`. Must be one of the following types:
    `float16`, `float32`, `float64`.

#### Returns:

A `Tensor`. Has the same type as `x`.


