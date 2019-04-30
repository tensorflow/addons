<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.seq2seq.SequenceLoss" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
</div>

# tfa.seq2seq.SequenceLoss

## Class `SequenceLoss`

Weighted cross-entropy loss for a sequence of logits.



### Aliases:

* Class `tfa.seq2seq.SequenceLoss`
* Class `tfa.seq2seq.loss.SequenceLoss`



Defined in [`seq2seq/loss.py`](https://github.com/tensorflow/addons/tree/r0.3/tensorflow_addons/seq2seq/loss.py).

<!-- Placeholder for "Used in" -->


<h2 id="__init__"><code>__init__</code></h2>

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





## Methods

<h3 id="__call__"><code>__call__</code></h3>

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





