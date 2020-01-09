<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.image.random_hsv_in_yiq" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.image.random_hsv_in_yiq

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/image/distort_image_ops.py#L28-L105">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



<!-- Equality marker -->
Adjust hue, saturation, value of an RGB image randomly in YIQ color

**Aliases**: `tfa.image.distort_image_ops.random_hsv_in_yiq`

``` python
tfa.image.random_hsv_in_yiq(
    image,
    max_delta_hue=0,
    lower_saturation=1,
    upper_saturation=1,
    lower_value=1,
    upper_value=1,
    seed=None,
    name=None
)
```



<!-- Placeholder for "Used in" -->
space.

Equivalent to `adjust_yiq_hsv()` but uses a `delta_h` randomly
picked in the interval `[-max_delta_hue, max_delta_hue]`, a
`scale_saturation` randomly picked in the interval
`[lower_saturation, upper_saturation]`, and a `scale_value`
randomly picked in the interval `[lower_saturation, upper_saturation]`.

#### Args:


* <b>`image`</b>: RGB image or images. Size of the last dimension must be 3.
* <b>`max_delta_hue`</b>: float. Maximum value for the random delta_hue. Passing 0
  disables adjusting hue.
* <b>`lower_saturation`</b>: float. Lower bound for the random scale_saturation.
* <b>`upper_saturation`</b>: float. Upper bound for the random scale_saturation.
* <b>`lower_value`</b>: float. Lower bound for the random scale_value.
* <b>`upper_value`</b>: float. Upper bound for the random scale_value.
* <b>`seed`</b>: An operation-specific seed. It will be used in conjunction
  with the graph-level seed to determine the real seeds that will be
  used in this operation. Please see the documentation of
  set_random_seed for its interaction with the graph-level random seed.
* <b>`name`</b>: A name for this operation (optional).


#### Returns:

3-D float tensor of shape `[height, width, channels]`.



#### Raises:


* <b>`ValueError`</b>: if `max_delta`, `lower_saturation`, `upper_saturation`,
  `lower_value`, or `upper_value` is invalid.

