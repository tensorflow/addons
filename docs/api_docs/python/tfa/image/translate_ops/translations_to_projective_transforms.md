<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.image.translate_ops.translations_to_projective_transforms" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.image.translate_ops.translations_to_projective_transforms

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/image/translate_ops.py#L25-L67">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



<!-- Equality marker -->
Returns projective transform(s) for the given translation(s).

``` python
tfa.image.translate_ops.translations_to_projective_transforms(
    translations,
    name=None
)
```



<!-- Placeholder for "Used in" -->


#### Args:


* <b>`translations`</b>: A 2-element list representing [dx, dy] or a matrix of
    2-element lists representing [dx, dy] to translate for each image
    (for a batch of images). The rank must be statically known
    (the shape is not `TensorShape(None)`).
* <b>`name`</b>: The name of the op.

#### Returns:

A tensor of shape (num_images, 8) projective transforms which can be
given to <a href="../../../tfa/image/transform.md"><code>tfa.image.transform</code></a>.


