<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.activations.gelu" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.activations.gelu

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/activations/gelu.py#L27-L47">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



<!-- Equality marker -->
Gaussian Error Linear Unit.

``` python
tfa.activations.gelu(
    x,
    approximate=True
)
```



<!-- Placeholder for "Used in" -->

Computes gaussian error linear:
`0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))` or
`x * P(X <= x) = 0.5 * x * (1 + erf(x / sqrt(2)))`, where P(X) ~ N(0, 1),
depending on whether approximation is enabled.

See [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415)
and [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805).

#### Args:


* <b>`x`</b>: A `Tensor`. Must be one of the following types:
    `float16`, `float32`, `float64`.
* <b>`approximate`</b>: bool, whether to enable approximation.

#### Returns:

A `Tensor`. Has the same type as `x`.


