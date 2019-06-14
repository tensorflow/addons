<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.image.mean_filter2d" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.image.mean_filter2d

Perform mean filtering on image(s).

### Aliases:

* `tfa.image.filters.mean_filter2d`
* `tfa.image.mean_filter2d`

``` python
tfa.image.mean_filter2d(
    image,
    filter_shape=(3, 3),
    padding='REFLECT',
    constant_values=0,
    name=None
)
```



Defined in [`image/filters.py`](https://github.com/tensorflow/addons/tree/0.4-release/tensorflow_addons/image/filters.py).

<!-- Placeholder for "Used in" -->


#### Args:


* <b>`image`</b>: Either a 3-D `Tensor` of shape `[height, width, channels]`,
  or a 4-D `Tensor` of shape `[batch_size, height, width, channels]`.
* <b>`filter_shape`</b>: An `integer` or `tuple`/`list` of 2 integers, specifying
  the height and width of the 2-D mean filter. Can be a single integer
  to specify the same value for all spatial dimensions.
* <b>`padding`</b>: A `string`, one of "REFLECT", "CONSTANT", or "SYMMETRIC".
  The type of padding algorithm to use, which is compatible with
  `mode` argument in `tf.pad`. For more details, please refer to
  https://www.tensorflow.org/api_docs/python/tf/pad.
* <b>`constant_values`</b>: A `scalar`, the pad value to use in "CONSTANT"
  padding mode.
* <b>`name`</b>: A name for this operation (optional).

#### Returns:

3-D or 4-D `Tensor` of the same dtype as input.


#### Raises:


* <b>`ValueError`</b>: If `image` is not 3 or 4-dimensional,
  if `padding` is other than "REFLECT", "CONSTANT" or "SYMMETRIC",
  or if `filter_shape` is invalid.