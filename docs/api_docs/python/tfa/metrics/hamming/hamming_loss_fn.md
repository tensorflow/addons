<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.metrics.hamming.hamming_loss_fn" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.metrics.hamming.hamming_loss_fn

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/metrics/hamming.py#L56-L130">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



<!-- Equality marker -->
Computes hamming loss.

``` python
tfa.metrics.hamming.hamming_loss_fn(
    y_true,
    y_pred,
    threshold,
    mode
)
```



<!-- Placeholder for "Used in" -->

Hamming loss is the fraction of wrong labels to the total number
of labels.

In multi-class classification, hamming loss is calculated as the
hamming distance between `actual` and `predictions`.
In multi-label classification, hamming loss penalizes only the
individual labels.

#### Args:



* <b>`y_true`</b>: actual target value
* <b>`y_pred`</b>: predicted target value
* <b>`threshold`</b>: Elements of `y_pred` greater than threshold are
    converted to be 1, and the rest 0. If threshold is
    None, the argmax is converted to 1, and the rest 0.
* <b>`mode`</b>: multi-class or multi-label


#### Returns:


hamming loss: float



#### Usage:



```python
# multi-class hamming loss
hl = HammingLoss(mode='multiclass', threshold=0.6)
actuals = tf.constant([[1, 0, 0, 0],[0, 0, 1, 0],
                   [0, 0, 0, 1],[0, 1, 0, 0]],
                  dtype=tf.float32)
predictions = tf.constant([[0.8, 0.1, 0.1, 0],
                           [0.2, 0, 0.8, 0],
                           [0.05, 0.05, 0.1, 0.8],
                           [1, 0, 0, 0]],
                      dtype=tf.float32)
hl.update_state(actuals, predictions)
print('Hamming loss: ', hl.result().numpy()) # 0.25

# multi-label hamming loss
hl = HammingLoss(mode='multilabel', threshold=0.8)
actuals = tf.constant([[1, 0, 1, 0],[0, 1, 0, 1],
                   [0, 0, 0,1]], dtype=tf.int32)
predictions = tf.constant([[0.82, 0.5, 0.90, 0],
                           [0, 1, 0.4, 0.98],
                           [0.89, 0.79, 0, 0.3]],
                           dtype=tf.float32)
hl.update_state(actuals, predictions)
print('Hamming loss: ', hl.result().numpy()) # 0.16666667
```

