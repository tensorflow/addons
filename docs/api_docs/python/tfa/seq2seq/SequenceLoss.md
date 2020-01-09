<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.seq2seq.SequenceLoss" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
</div>

# tfa.seq2seq.SequenceLoss

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/seq2seq/loss.py#L168-L206">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



<!-- Equality marker -->
## Class `SequenceLoss`

Weighted cross-entropy loss for a sequence of logits.



**Aliases**: `tfa.seq2seq.loss.SequenceLoss`

<!-- Placeholder for "Used in" -->


<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/seq2seq/loss.py#L171-L188">View source</a>

``` python
__init__(
    average_across_timesteps=False,
    average_across_batch=False,
    sum_over_timesteps=True,
    sum_over_batch=True,
    softmax_loss_function=None,
    name=None
)
```

Initialize self.  See help(type(self)) for accurate signature.




## Methods

<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/seq2seq/loss.py#L190-L202">View source</a>

``` python
__call__(
    y_true,
    y_pred,
    sample_weight=None
)
```

Override the parent __call__ to have a customized reduce
behavior.

<h3 id="from_config"><code>from_config</code></h3>

``` python
from_config(
    cls,
    config
)
```

Instantiates a `Loss` from its config (output of `get_config()`).


#### Args:


* <b>`config`</b>: Output of `get_config()`.


#### Returns:

A `Loss` instance.


<h3 id="get_config"><code>get_config</code></h3>

``` python
get_config()
```








