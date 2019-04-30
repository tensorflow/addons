<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.image.euclidean_dist_transform" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.image.euclidean_dist_transform

Applies euclidean distance transform(s) to the image(s).

### Aliases:

* `tfa.image.distance_transform.euclidean_dist_transform`
* `tfa.image.euclidean_dist_transform`

``` python
tfa.image.euclidean_dist_transform(
    images,
    dtype=tf.float32,
    name=None
)
```



Defined in [`image/distance_transform.py`](https://github.com/tensorflow/addons/tree/r0.3/tensorflow_addons/image/distance_transform.py).

<!-- Placeholder for "Used in" -->

#### Args:

* <b>`images`</b>: A tensor of shape (num_images, num_rows, num_columns, 1) (NHWC),
    or (num_rows, num_columns, 1) (HWC). The rank must be statically known
    (the shape is not `TensorShape(None)`.
* <b>`dtype`</b>: DType of the output tensor.
* <b>`name`</b>: The name of the op.


#### Returns:

Image(s) with the type `dtype` and same shape as `images`, with the
transform applied. If a tensor of all ones is given as input, the
output tensor will be filled with the max value of the `dtype`.


#### Raises:

* <b>`TypeError`</b>: If `image` is not tf.uint8, or `dtype` is not floating point.
* <b>`ValueError`</b>: If `image` more than one channel, or `image` is not of
    rank 3 or 4.