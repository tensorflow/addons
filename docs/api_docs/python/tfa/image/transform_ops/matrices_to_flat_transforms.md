<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.image.transform_ops.matrices_to_flat_transforms" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.image.transform_ops.matrices_to_flat_transforms

Converts affine matrices to projective transforms.

``` python
tfa.image.transform_ops.matrices_to_flat_transforms(
    transform_matrices,
    name=None
)
```



Defined in [`image/transform_ops.py`](https://github.com/tensorflow/addons/tree/r0.3/tensorflow_addons/image/transform_ops.py).

<!-- Placeholder for "Used in" -->

Note that we expect matrices that map output coordinates to input
coordinates. To convert forward transformation matrices,
call `tf.linalg.inv` on the matrices and use the result here.

#### Args:

* <b>`transform_matrices`</b>: One or more affine transformation matrices, for the
    reverse transformation in homogeneous coordinates. Shape `(3, 3)` or
    `(N, 3, 3)`.
* <b>`name`</b>: The name for the op.


#### Returns:

2D tensor of flat transforms with shape `(N, 8)`, which may be passed
into `transform` op.


#### Raises:

* <b>`ValueError`</b>: If `transform_matrices` have an invalid shape.