<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.metrics.hamming_distance" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.metrics.hamming_distance

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/metrics/hamming.py#L25-L53">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



<!-- Equality marker -->
Computes hamming distance.

**Aliases**: `tfa.metrics.hamming.hamming_distance`

``` python
tfa.metrics.hamming_distance(
    actuals,
    predictions
)
```



<!-- Placeholder for "Used in" -->

Hamming distance is for comparing two binary strings.
It is the number of bit positions in which two bits
are different.

#### Args:


* <b>`actuals`</b>: actual target value
* <b>`predictions`</b>: predicted value


#### Returns:

hamming distance: float



#### Usage:



```python
actuals = tf.constant([1, 1, 0, 0, 1, 0, 1, 0, 0, 1],
                      dtype=tf.int32)
predictions = tf.constant([1, 0, 0, 0, 1, 0, 0, 1, 0, 1],
                          dtype=tf.int32)
result = hamming_distance(actuals, predictions)
print('Hamming distance: ', result.numpy())
```

