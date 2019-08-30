<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.image.transform_ops.compose_transforms" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.image.transform_ops.compose_transforms


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.5/tensorflow_addons/image/transform_ops.py#L107-L129">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Composes the transforms tensors.

``` python
tfa.image.transform_ops.compose_transforms(
    transforms,
    name=None
)
```



<!-- Placeholder for "Used in" -->


#### Args:


* <b>`transforms`</b>: List of image projective transforms to be composed. Each
  transform is length 8 (single transform) or shape (N, 8) (batched
  transforms). The shapes of all inputs must be equal, and at least one
  input must be given.
* <b>`name`</b>: The name for the op.


#### Returns:

A composed transform tensor. When passed to `transform` op,
    equivalent to applying each of the given transforms to the image in
    order.
