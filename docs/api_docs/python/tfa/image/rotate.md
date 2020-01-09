<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.image.rotate" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.image.rotate

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/image/transform_ops.py#L281-L316">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



<!-- Equality marker -->
Rotate image(s) counterclockwise by the passed angle(s) in radians.

**Aliases**: `tfa.image.transform_ops.rotate`

``` python
tfa.image.rotate(
    images,
    angles,
    interpolation='NEAREST',
    name=None
)
```



<!-- Placeholder for "Used in" -->


#### Args:


* <b>`images`</b>: A tensor of shape
  (num_images, num_rows, num_columns, num_channels)
  (NHWC), (num_rows, num_columns, num_channels) (HWC), or
  (num_rows, num_columns) (HW).
* <b>`angles`</b>: A scalar angle to rotate all images by, or (if images has rank 4)
  a vector of length num_images, with an angle for each image in the
  batch.
* <b>`interpolation`</b>: Interpolation mode. Supported values: "NEAREST",
  "BILINEAR".
* <b>`name`</b>: The name of the op.


#### Returns:

Image(s) with the same type and shape as `images`, rotated by the given
angle(s). Empty space due to the rotation will be filled with zeros.



#### Raises:


* <b>`TypeError`</b>: If `image` is an invalid type.

