<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.optimizers" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfa.optimizers


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.5/tensorflow_addons/optimizers/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Additional optimizers that conform to Keras API.

<!-- Placeholder for "Used in" -->


## Modules

[`lazy_adam`](../tfa/optimizers/lazy_adam.md) module: Variant of the Adam optimizer that handles sparse updates more efficiently.

[`moving_average`](../tfa/optimizers/moving_average.md) module

[`weight_decay_optimizers`](../tfa/optimizers/weight_decay_optimizers.md) module: Base class to make optimizers weight decay ready.

## Classes

[`class AdamW`](../tfa/optimizers/AdamW.md): Optimizer that implements the Adam algorithm with weight decay.

[`class LazyAdam`](../tfa/optimizers/LazyAdam.md): Variant of the Adam optimizer that handles sparse updates more

[`class MovingAverage`](../tfa/optimizers/MovingAverage.md): Optimizer that computes a moving average of the variables.

[`class SGDW`](../tfa/optimizers/SGDW.md): Optimizer that implements the Momentum algorithm with weight_decay.

## Functions

[`extend_with_decoupled_weight_decay(...)`](../tfa/optimizers/extend_with_decoupled_weight_decay.md): Factory function returning an optimizer class with decoupled weight

