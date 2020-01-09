<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.activations.mish" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.activations.mish

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/activations/mish.py#L27-L42">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



<!-- Equality marker -->
Mish: A Self Regularized Non-Monotonic Neural Activation Function.

``` python
tfa.activations.mish(x)
```



<!-- Placeholder for "Used in" -->

Computes mish activation: x * tanh(softplus(x))

See [Mish: A Self Regularized Non-Monotonic Neural Activation Function](https://arxiv.org/abs/1908.08681).

#### Args:


* <b>`x`</b>: A `Tensor`. Must be one of the following types:
    `float16`, `float32`, `float64`.

#### Returns:

A `Tensor`. Has the same type as `x`.


