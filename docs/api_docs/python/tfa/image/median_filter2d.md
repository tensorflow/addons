<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.image.median_filter2d" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.image.median_filter2d

This method performs Median Filtering on image. Filter shape can be user

### Aliases:

* `tfa.image.filters.median_filter2d`
* `tfa.image.median_filter2d`

``` python
tfa.image.median_filter2d(
    image,
    filter_shape=(3, 3),
    name=None
)
```



Defined in [`image/filters.py`](https://github.com/tensorflow/addons/tree/r0.3/tensorflow_addons/image/filters.py).

<!-- Placeholder for "Used in" -->
given.

This method takes both kind of images where pixel values lie between 0 to
255 and where it lies between 0.0 and 1.0
#### Args:

* <b>`image`</b>: A 3D `Tensor` of type `float32` or 'int32' or 'float64' or
           'int64 and of shape`[rows, columns, channels]`

* <b>`filter_shape`</b>: Optional Argument. A tuple of 2 integers (R,C).
           R is the first value is the number of rows in the filter and
           C is the second value in the filter is the number of columns
           in the filter. This creates a filter of shape (R,C) or RxC
           filter. Default value = (3,3)
* <b>`name`</b>: The name of the op.

 Returns:
     A 3D median filtered image tensor of shape [rows,columns,channels] and
     type 'int32'. Pixel value of returned tensor ranges between 0 to 255