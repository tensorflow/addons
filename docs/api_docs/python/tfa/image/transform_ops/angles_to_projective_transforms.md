<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.image.transform_ops.angles_to_projective_transforms" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.image.transform_ops.angles_to_projective_transforms

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/image/transform_ops.py#L199-L244">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



<!-- Equality marker -->
Returns projective transform(s) for the given angle(s).

``` python
tfa.image.transform_ops.angles_to_projective_transforms(
    angles,
    image_height,
    image_width,
    name=None
)
```



<!-- Placeholder for "Used in" -->


#### Args:


* <b>`angles`</b>: A scalar angle to rotate all images by, or (for batches of
  images) a vector with an angle to rotate each image in the batch. The
  rank must be statically known (the shape is not `TensorShape(None)`.
* <b>`image_height`</b>: Height of the image(s) to be transformed.
* <b>`image_width`</b>: Width of the image(s) to be transformed.


#### Returns:

A tensor of shape (num_images, 8). Projective transforms which can be
given to `transform` op.


