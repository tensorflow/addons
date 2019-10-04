<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.activations.tanhshrink" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.activations.tanhshrink


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.6/tensorflow_addons/activations/tanhshrink.py#L28-L40">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Applies the element-wise function: x - tanh(x)

``` python
tfa.activations.tanhshrink(x)
```



<!-- Placeholder for "Used in" -->


#### Args:


* <b>`features`</b>: A `Tensor`. Must be one of the following types:
    `float16`, `float32`, `float64`.

#### Returns:

A `Tensor`. Has the same type as `features`.
