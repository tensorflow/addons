<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.image.sparse_image_warp" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.image.sparse_image_warp

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/image/sparse_image_warp.py#L100-L200">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



<!-- Equality marker -->
Image warping using correspondences between sparse control points.

``` python
tfa.image.sparse_image_warp(
    image,
    source_control_point_locations,
    dest_control_point_locations,
    interpolation_order=2,
    regularization_weight=0.0,
    num_boundary_points=0,
    name='sparse_image_warp'
)
```



<!-- Placeholder for "Used in" -->

Apply a non-linear warp to the image, where the warp is specified by
the source and destination locations of a (potentially small) number of
control points. First, we use a polyharmonic spline
(`tf.contrib.image.interpolate_spline`) to interpolate the displacements
between the corresponding control points to a dense flow field.
Then, we warp the image using this dense flow field
(`tf.contrib.image.dense_image_warp`).

Let t index our control points. For regularization_weight=0, we have:
warped_image[b, dest_control_point_locations[b, t, 0],
                dest_control_point_locations[b, t, 1], :] =
image[b, source_control_point_locations[b, t, 0],
         source_control_point_locations[b, t, 1], :].

For regularization_weight > 0, this condition is met approximately, since
regularized interpolation trades off smoothness of the interpolant vs.
reconstruction of the interpolant at the control points.
See `tf.contrib.image.interpolate_spline` for further documentation of the
interpolation_order and regularization_weight arguments.


#### Args:


* <b>`image`</b>: `[batch, height, width, channels]` float `Tensor`
* <b>`source_control_point_locations`</b>: `[batch, num_control_points, 2]` float
  `Tensor`
* <b>`dest_control_point_locations`</b>: `[batch, num_control_points, 2]` float
  `Tensor`
* <b>`interpolation_order`</b>: polynomial order used by the spline interpolation
* <b>`regularization_weight`</b>: weight on smoothness regularizer in interpolation
* <b>`num_boundary_points`</b>: How many zero-flow boundary points to include at
  each image edge.Usage:
    num_boundary_points=0: don't add zero-flow points
    num_boundary_points=1: 4 corners of the image
    num_boundary_points=2: 4 corners and one in the middle of each edge
      (8 points total)
    num_boundary_points=n: 4 corners and n-1 along each edge
* <b>`name`</b>: A name for the operation (optional).

Note that image and offsets can be of type tf.half, tf.float32, or
tf.float64, and do not necessarily have to be the same type.


#### Returns:


* <b>`warped_image`</b>: `[batch, height, width, channels]` float `Tensor` with same
  type as input image.
* <b>`flow_field`</b>: `[batch, height, width, 2]` float `Tensor` containing the
  dense flow field produced by the interpolation.

