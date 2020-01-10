<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.optimizers.stochastic_weight_averaging" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfa.optimizers.stochastic_weight_averaging


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/optimizers/stochastic_weight_averaging.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



An implementation of the Stochastic Weight Averaging optimizer.


The Stochastic Weight Averaging mechanism was proposed by Pavel Izmailov
et. al in the paper [Averaging Weights Leads to Wider Optima and Better
Generalization](https://arxiv.org/abs/1803.05407). The optimizer
implements averaging of multiple points along the trajectory of SGD.
This averaging has shown to improve model performance on validation/test
sets whilst possibly causing a small increase in loss on the training
set.

## Classes

[`class SWA`](../../tfa/optimizers/SWA.md): This class extends optimizers with Stochastic Weight Averaging (SWA).



