<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.optimizers.RectifiedAdam" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="iterations"/>
<meta itemprop="property" content="weights"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="add_slot"/>
<meta itemprop="property" content="add_weight"/>
<meta itemprop="property" content="apply_gradients"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
<meta itemprop="property" content="get_gradients"/>
<meta itemprop="property" content="get_slot"/>
<meta itemprop="property" content="get_slot_names"/>
<meta itemprop="property" content="get_updates"/>
<meta itemprop="property" content="get_weights"/>
<meta itemprop="property" content="minimize"/>
<meta itemprop="property" content="set_weights"/>
<meta itemprop="property" content="variables"/>
</div>

# tfa.optimizers.RectifiedAdam

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/optimizers/rectified_adam.py#L24-L306">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



<!-- Equality marker -->
## Class `RectifiedAdam`

Variant of the Adam optimizer whose adaptive learning rate is rectified



**Aliases**: `tfa.optimizers.rectified_adam.RectifiedAdam`

<!-- Placeholder for "Used in" -->
so as to have a consistent variance.

It implements the Rectified Adam (a.k.a. RAdam) proposed by
Liyuan Liu et al. in [On The Variance Of The Adaptive Learning Rate
And Beyond](https://arxiv.org/pdf/1908.03265v1.pdf).

#### Example of usage:



```python
opt = tfa.optimizers.RectifiedAdam(lr=1e-3)
```

Note: `amsgrad` is not described in the original paper. Use it with
      caution.

RAdam is not a placement of the heuristic warmup, the settings should be
kept if warmup has already been employed and tuned in the baseline method.
You can enable warmup by setting `total_steps` and `warmup_proportion`:

```python
opt = tfa.optimizers.RectifiedAdam(
    lr=1e-3,
    total_steps=10000,
    warmup_proportion=0.1,
    min_lr=1e-5,
)
```

In the above example, the learning rate will increase linearly
from 0 to `lr` in 1000 steps, then decrease linearly from `lr` to `min_lr`
in 9000 steps.

Lookahead, proposed by Michael R. Zhang et.al in the paper
[Lookahead Optimizer: k steps forward, 1 step back]
(https://arxiv.org/abs/1907.08610v1), can be integrated with RAdam,
which is announced by Less Wright and the new combined optimizer can also
be called "Ranger". The mechanism can be enabled by using the lookahead
wrapper. For example:

```python
radam = tfa.optimizers.RectifiedAdam()
ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
```

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/optimizers/rectified_adam.py#L71-L128">View source</a>

``` python
__init__(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    weight_decay=0.0,
    amsgrad=False,
    sma_threshold=5.0,
    total_steps=0,
    warmup_proportion=0.1,
    min_lr=0.0,
    name='RectifiedAdam',
    **kwargs
)
```

Construct a new RAdam optimizer.


#### Args:


* <b>`learning_rate`</b>: A `Tensor` or a floating point value. or a schedule
    that is a `tf.keras.optimizers.schedules.LearningRateSchedule`
    The learning rate.
* <b>`beta_1`</b>: A float value or a constant float tensor.
    The exponential decay rate for the 1st moment estimates.
* <b>`beta_2`</b>: A float value or a constant float tensor.
    The exponential decay rate for the 2nd moment estimates.
* <b>`epsilon`</b>: A small constant for numerical stability.
* <b>`weight_decay`</b>: A floating point value. Weight decay for each param.
* <b>`amsgrad`</b>: boolean. Whether to apply AMSGrad variant of this
    algorithm from the paper "On the Convergence of Adam and
    beyond".
sma_threshold. A float value.
    The threshold for simple mean average.
* <b>`total_steps`</b>: An integer. Total number of training steps.
    Enable warmup by setting a positive value.
* <b>`warmup_proportion`</b>: A floating point value.
    The proportion of increasing steps.
* <b>`min_lr`</b>: A floating point value. Minimum learning rate after warmup.
* <b>`name`</b>: Optional name for the operations created when applying
    gradients. Defaults to "RectifiedAdam".
* <b>`**kwargs`</b>: keyword arguments. Allowed to be {`clipnorm`,
    `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients
    by norm; `clipvalue` is clip gradients by value, `decay` is
    included for backward compatibility to allow time inverse
    decay of learning rate. `lr` is included for backward
    compatibility, recommended to use `learning_rate` instead.



## Properties

<h3 id="iterations"><code>iterations</code></h3>

Variable. The number of training steps this Optimizer has run.


<h3 id="weights"><code>weights</code></h3>

Returns variables of this Optimizer based on the order created.




## Methods

<h3 id="add_slot"><code>add_slot</code></h3>

``` python
add_slot(
    var,
    slot_name,
    initializer='zeros'
)
```

Add a new slot variable for `var`.


<h3 id="add_weight"><code>add_weight</code></h3>

``` python
add_weight(
    name,
    shape,
    dtype=None,
    initializer='zeros',
    trainable=None,
    synchronization=tf_variables.VariableSynchronization.AUTO,
    aggregation=tf_variables.VariableAggregation.NONE
)
```




<h3 id="apply_gradients"><code>apply_gradients</code></h3>

``` python
apply_gradients(
    grads_and_vars,
    name=None
)
```

Apply gradients to variables.

This is the second part of `minimize()`. It returns an `Operation` that
applies gradients.

#### Args:


* <b>`grads_and_vars`</b>: List of (gradient, variable) pairs.
* <b>`name`</b>: Optional name for the returned operation.  Default to the name
  passed to the `Optimizer` constructor.


#### Returns:

An `Operation` that applies the specified gradients. The `iterations`
will be automatically increased by 1.



#### Raises:


* <b>`TypeError`</b>: If `grads_and_vars` is malformed.
* <b>`ValueError`</b>: If none of the variables have gradients.

<h3 id="from_config"><code>from_config</code></h3>

``` python
from_config(
    cls,
    config,
    custom_objects=None
)
```

Creates an optimizer from its config.

This method is the reverse of `get_config`,
capable of instantiating the same optimizer from the config
dictionary.

#### Arguments:


* <b>`config`</b>: A Python dictionary, typically the output of get_config.
* <b>`custom_objects`</b>: A Python dictionary mapping names to additional Python
  objects used to create this optimizer, such as a function used for a
  hyperparameter.


#### Returns:

An optimizer instance.


<h3 id="get_config"><code>get_config</code></h3>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/optimizers/rectified_adam.py#L280-L306">View source</a>

``` python
get_config()
```

Returns the config of the optimimizer.

An optimizer config is a Python dictionary (serializable)
containing the configuration of an optimizer.
The same optimizer can be reinstantiated later
(without any saved state) from this configuration.

#### Returns:

Python dictionary.


<h3 id="get_gradients"><code>get_gradients</code></h3>

``` python
get_gradients(
    loss,
    params
)
```

Returns gradients of `loss` with respect to `params`.


#### Arguments:


* <b>`loss`</b>: Loss tensor.
* <b>`params`</b>: List of variables.


#### Returns:

List of gradient tensors.



#### Raises:


* <b>`ValueError`</b>: In case any gradient cannot be computed (e.g. if gradient
  function not implemented).

<h3 id="get_slot"><code>get_slot</code></h3>

``` python
get_slot(
    var,
    slot_name
)
```




<h3 id="get_slot_names"><code>get_slot_names</code></h3>

``` python
get_slot_names()
```

A list of names for this optimizer's slots.


<h3 id="get_updates"><code>get_updates</code></h3>

``` python
get_updates(
    loss,
    params
)
```




<h3 id="get_weights"><code>get_weights</code></h3>

``` python
get_weights()
```




<h3 id="minimize"><code>minimize</code></h3>

``` python
minimize(
    loss,
    var_list,
    grad_loss=None,
    name=None
)
```

Minimize `loss` by updating `var_list`.

This method simply computes gradient using `tf.GradientTape` and calls
`apply_gradients()`. If you want to process the gradient before applying
then call `tf.GradientTape` and `apply_gradients()` explicitly instead
of using this function.

#### Args:


* <b>`loss`</b>: A callable taking no arguments which returns the value to minimize.
* <b>`var_list`</b>: list or tuple of `Variable` objects to update to minimize
  `loss`, or a callable returning the list or tuple of `Variable` objects.
  Use callable when the variable list would otherwise be incomplete before
  `minimize` since the variables are created at the first time `loss` is
  called.
* <b>`grad_loss`</b>: Optional. A `Tensor` holding the gradient computed for `loss`.
* <b>`name`</b>: Optional name for the returned operation.


#### Returns:

An `Operation` that updates the variables in `var_list`. The `iterations`
will be automatically increased by 1.



#### Raises:


* <b>`ValueError`</b>: If some of the variables are not `Variable` objects.

<h3 id="set_weights"><code>set_weights</code></h3>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/optimizers/rectified_adam.py#L139-L144">View source</a>

``` python
set_weights(weights)
```




<h3 id="variables"><code>variables</code></h3>

``` python
variables()
```

Returns variables of this Optimizer based on the order created.






