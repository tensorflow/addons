<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.optimizers.Triangular2CyclicalLearningRate" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
</div>

# tfa.optimizers.Triangular2CyclicalLearningRate

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/optimizers/cyclical_learning_rate.py#L171-L228">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



<!-- Equality marker -->
## Class `Triangular2CyclicalLearningRate`

A LearningRateSchedule that uses cyclical schedule.

Inherits From: [`CyclicalLearningRate`](../../tfa/optimizers/CyclicalLearningRate.md)

**Aliases**: `tfa.optimizers.cyclical_learning_rate.Triangular2CyclicalLearningRate`

<!-- Placeholder for "Used in" -->


<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/optimizers/cyclical_learning_rate.py#L172-L228">View source</a>

``` python
__init__(
    initial_learning_rate,
    maximal_learning_rate,
    step_size,
    scale_mode='cycle',
    name='Triangular2CyclicalLearningRate'
)
```

Applies triangular2 cyclical schedule to the learning rate.

See Cyclical Learning Rates for Training Neural Networks. https://arxiv.org/abs/1506.01186


```python
from tf.keras.optimizers import schedules

lr_schedule = schedules.Triangular2CyclicalLearningRate(
    initial_learning_rate=1e-4,
    maximal_learning_rate=1e-2,
    step_size=2000,
    scale_mode="cycle",
    name="MyCyclicScheduler")

model.compile(optimizer=tf.keras.optimizers.SGD(
                                            learning_rate=lr_schedule),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(data, labels, epochs=5)
```

You can pass this schedule directly into a
`tf.keras.optimizers.Optimizer` as the learning rate.

#### Args:


* <b>`initial_learning_rate`</b>: A scalar `float32` or `float64` `Tensor` or
    a Python number.  The initial learning rate.
* <b>`maximal_learning_rate`</b>: A scalar `float32` or `float64` `Tensor` or
    a Python number.  The maximum learning rate.
* <b>`step_size`</b>: A scalar `float32` or `float64` `Tensor` or a
    Python number. Step size.
* <b>`scale_fn`</b>: A function. Scheduling function applied in cycle
* <b>`scale_mode`</b>: ['cycle', 'iterations']. Mode to apply during cyclic
    schedule
* <b>`name`</b>: (Optional) Name for the operation.


#### Returns:

Updated learning rate value.




## Methods

<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/optimizers/cyclical_learning_rate.py#L84-L98">View source</a>

``` python
__call__(step)
```

Call self as a function.


<h3 id="from_config"><code>from_config</code></h3>

``` python
from_config(
    cls,
    config
)
```

Instantiates a `LearningRateSchedule` from its config.


#### Args:


* <b>`config`</b>: Output of `get_config()`.


#### Returns:

A `LearningRateSchedule` instance.


<h3 id="get_config"><code>get_config</code></h3>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/optimizers/cyclical_learning_rate.py#L100-L106">View source</a>

``` python
get_config()
```








