<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.image.euclidean_dist_transform" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.image.euclidean_dist_transform


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.5/tensorflow_addons/image/distance_transform.py#L30-L70">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Applies euclidean distance transform(s) to the image(s).

### Aliases:

* `tfa.image.distance_transform.euclidean_dist_transform`


``` python
tfa.image.euclidean_dist_transform(
    images,
    dtype=tf.float32,
    name=None
)
```



<!-- Placeholder for "Used in" -->


#### Args:


* <b>`images`</b>: A tensor of shape (num_images, num_rows, num_columns, 1) (NHWC),
  or (num_rows, num_columns, 1) (HWC) or (num_rows, num_columns) (HW).
* <b>`dtype`</b>: DType of the output tensor.
* <b>`name`</b>: The name of the op.


#### Returns:

Image(s) with the type `dtype` and same shape as `images`, with the
transform applied. If a tensor of all ones is given as input, the
output tensor will be filled with the max value of the `dtype`.



#### Raises:


* <b>`TypeError`</b>: If `image` is not tf.uint8, or `dtype` is not floating point.
* <b>`ValueError`</b>: If `image` more than one channel, or `image` is not of
  rank between 2 and 4.