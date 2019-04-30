<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.seq2seq.ScheduledOutputTrainingSampler" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="batch_size"/>
<meta itemprop="property" content="sample_ids_dtype"/>
<meta itemprop="property" content="sample_ids_shape"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="initialize"/>
<meta itemprop="property" content="next_inputs"/>
<meta itemprop="property" content="sample"/>
</div>

# tfa.seq2seq.ScheduledOutputTrainingSampler

## Class `ScheduledOutputTrainingSampler`

A training sampler that adds scheduled sampling directly to outputs.

Inherits From: [`TrainingSampler`](../../tfa/seq2seq/TrainingSampler.md)

### Aliases:

* Class `tfa.seq2seq.ScheduledOutputTrainingSampler`
* Class `tfa.seq2seq.sampler.ScheduledOutputTrainingSampler`



Defined in [`seq2seq/sampler.py`](https://github.com/tensorflow/addons/tree/r0.3/tensorflow_addons/seq2seq/sampler.py).

<!-- Placeholder for "Used in" -->

Returns False for sample_ids where no sampling took place; True
elsewhere.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    sampling_probability,
    time_major=False,
    seed=None,
    next_inputs_fn=None
)
```

Initializer.

#### Args:

* <b>`sampling_probability`</b>: A `float32` scalar tensor: the probability of
    sampling from the outputs instead of reading directly from the
    inputs.
* <b>`time_major`</b>: Python bool. Whether the tensors in `inputs` are time
    major. If `False` (default), they are assumed to be batch major.
* <b>`seed`</b>: The sampling seed.
* <b>`next_inputs_fn`</b>: (Optional) callable to apply to the RNN outputs to
    create the next input when sampling. If `None` (default), the RNN
    outputs will be used as the next inputs.


#### Raises:

* <b>`ValueError`</b>: if `sampling_probability` is not a scalar or vector.



## Properties

<h3 id="batch_size"><code>batch_size</code></h3>



<h3 id="sample_ids_dtype"><code>sample_ids_dtype</code></h3>



<h3 id="sample_ids_shape"><code>sample_ids_shape</code></h3>





## Methods

<h3 id="initialize"><code>initialize</code></h3>

``` python
initialize(
    inputs,
    sequence_length=None,
    auxiliary_inputs=None
)
```



<h3 id="next_inputs"><code>next_inputs</code></h3>

``` python
next_inputs(
    time,
    outputs,
    state,
    sample_ids
)
```



<h3 id="sample"><code>sample</code></h3>

``` python
sample(
    time,
    outputs,
    state
)
```





