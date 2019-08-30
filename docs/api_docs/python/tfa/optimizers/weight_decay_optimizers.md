<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.optimizers.weight_decay_optimizers" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfa.optimizers.weight_decay_optimizers


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.5/tensorflow_addons/optimizers/weight_decay_optimizers.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Base class to make optimizers weight decay ready.

<!-- Placeholder for "Used in" -->


## Classes

[`class AdamW`](../../tfa/optimizers/AdamW.md): Optimizer that implements the Adam algorithm with weight decay.

[`class DecoupledWeightDecayExtension`](../../tfa/optimizers/weight_decay_optimizers/DecoupledWeightDecayExtension.md): This class allows to extend optimizers with decoupled weight decay.

[`class SGDW`](../../tfa/optimizers/SGDW.md): Optimizer that implements the Momentum algorithm with weight_decay.

## Functions

[`extend_with_decoupled_weight_decay(...)`](../../tfa/optimizers/extend_with_decoupled_weight_decay.md): Factory function returning an optimizer class with decoupled weight

