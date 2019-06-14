<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.image.transform_ops.flat_transforms_to_matrices" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.image.transform_ops.flat_transforms_to_matrices

Converts projective transforms to affine matrices.

``` python
tfa.image.transform_ops.flat_transforms_to_matrices(
    transforms,
    name=None
)
```



Defined in [`image/transform_ops.py`](https://github.com/tensorflow/addons/tree/0.4-release/tensorflow_addons/image/transform_ops.py).

<!-- Placeholder for "Used in" -->

Note that the output matrices map output coordinates to input coordinates.
For the forward transformation matrix, call `tf.linalg.inv` on the result.

#### Args:


* <b>`transforms`</b>: Vector of length 8, or batches of transforms with shape
  `(N, 8)`.
* <b>`name`</b>: The name for the op.


#### Returns:

3D tensor of matrices with shape `(N, 3, 3)`. The output matrices map the
  *output coordinates* (in homogeneous coordinates) of each transform to
  the corresponding *input coordinates*.



#### Raises:


* <b>`ValueError`</b>: If `transforms` have an invalid shape.