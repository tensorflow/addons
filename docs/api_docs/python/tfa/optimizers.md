<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.optimizers" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfa.optimizers


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.6/tensorflow_addons/optimizers/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Additional optimizers that conform to Keras API.

<!-- Placeholder for "Used in" -->


## Modules

[`conditional_gradient`](../tfa/optimizers/conditional_gradient.md) module: Conditional Gradient method for TensorFlow.

[`lazy_adam`](../tfa/optimizers/lazy_adam.md) module: Variant of the Adam optimizer that handles sparse updates more efficiently.

[`lookahead`](../tfa/optimizers/lookahead.md) module

[`moving_average`](../tfa/optimizers/moving_average.md) module

[`rectified_adam`](../tfa/optimizers/rectified_adam.md) module: Rectified Adam (RAdam) optimizer.

[`weight_decay_optimizers`](../tfa/optimizers/weight_decay_optimizers.md) module: Base class to make optimizers weight decay ready.

## Classes

[`class AdamW`](../tfa/optimizers/AdamW.md): Optimizer that implements the Adam algorithm with weight decay.

[`class ConditionalGradient`](../tfa/optimizers/ConditionalGradient.md): Optimizer that implements the Conditional Gradient optimization.

[`class LazyAdam`](../tfa/optimizers/LazyAdam.md): Variant of the Adam optimizer that handles sparse updates more

[`class Lookahead`](../tfa/optimizers/Lookahead.md): This class allows to extend optimizers with the lookahead mechanism.

[`class MovingAverage`](../tfa/optimizers/MovingAverage.md): Optimizer that computes a moving average of the variables.

[`class RectifiedAdam`](../tfa/optimizers/RectifiedAdam.md): Variant of the Adam optimizer whose adaptive learning rate is rectified

[`class SGDW`](../tfa/optimizers/SGDW.md): Optimizer that implements the Momentum algorithm with weight_decay.

## Functions

[`extend_with_decoupled_weight_decay(...)`](../tfa/optimizers/extend_with_decoupled_weight_decay.md): Factory function returning an optimizer class with decoupled weight

