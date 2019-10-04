<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.image.adjust_hsv_in_yiq" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.image.adjust_hsv_in_yiq


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.6/tensorflow_addons/image/distort_image_ops.py#L109-L149">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Adjust hue, saturation, value of an RGB image in YIQ color space.

### Aliases:

* `tfa.image.distort_image_ops.adjust_hsv_in_yiq`


``` python
tfa.image.adjust_hsv_in_yiq(
    image,
    delta_hue=0,
    scale_saturation=1,
    scale_value=1,
    name=None
)
```



<!-- Placeholder for "Used in" -->

This is a convenience method that converts an RGB image to float
representation, converts it to YIQ, rotates the color around the
Y channel by delta_hue in radians, scales the chrominance channels
(I, Q) by scale_saturation, scales all channels (Y, I, Q) by scale_value,
converts back to RGB, and then back to the original data type.

`image` is an RGB image. The image hue is adjusted by converting the
image to YIQ, rotating around the luminance channel (Y) by
`delta_hue` in radians, multiplying the chrominance channels (I, Q) by
`scale_saturation`, and multiplying all channels (Y, I, Q) by
`scale_value`. The image is then converted back to RGB.

#### Args:


* <b>`image`</b>: RGB image or images. Size of the last dimension must be 3.
* <b>`delta_hue`</b>: float, the hue rotation amount, in radians.
* <b>`scale_saturation`</b>: float, factor to multiply the saturation by.
* <b>`scale_value`</b>: float, factor to multiply the value by.
* <b>`name`</b>: A name for this operation (optional).


#### Returns:

Adjusted image(s), same shape and dtype as `image`.
