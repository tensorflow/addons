<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.activations.rrelu" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.activations.rrelu

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/activations/rrelu.py#L23-L56">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



<!-- Equality marker -->
rrelu function.

``` python
tfa.activations.rrelu(
    x,
    lower=0.125,
    upper=0.3333333333333333,
    training=None,
    seed=None
)
```



<!-- Placeholder for "Used in" -->

Computes rrelu function:
`x if x > 0 else random(lower, upper) * x` or
`x if x > 0 else x * (lower + upper) / 2`
depending on whether training is enabled.

See [Empirical Evaluation of Rectified Activations in Convolutional Network](https://arxiv.org/abs/1505.00853).

#### Args:


* <b>`x`</b>: A `Tensor`. Must be one of the following types:
    `float16`, `float32`, `float64`.
* <b>`lower`</b>: `float`, lower bound for random alpha.
* <b>`upper`</b>: `float`, upper bound for random alpha.
* <b>`training`</b>: `bool`, indicating whether the `call`
is meant for training or inference.
* <b>`seed`</b>: `int`, this sets the operation-level seed.

#### Returns:


* <b>`result`</b>: A `Tensor`. Has the same type as `x`.

