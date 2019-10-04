<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.losses" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfa.losses


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.6/tensorflow_addons/losses/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Additional losses that conform to Keras API.

<!-- Placeholder for "Used in" -->


## Modules

[`contrastive`](../tfa/losses/contrastive.md) module: Implements contrastive loss.

[`focal_loss`](../tfa/losses/focal_loss.md) module: Implements Focal loss.

[`lifted`](../tfa/losses/lifted.md) module: Implements lifted_struct_loss.

[`metric_learning`](../tfa/losses/metric_learning.md) module: Functions of metric learning.

[`npairs`](../tfa/losses/npairs.md) module: Implements npairs loss.

[`triplet`](../tfa/losses/triplet.md) module: Implements triplet loss.

## Classes

[`class ContrastiveLoss`](../tfa/losses/ContrastiveLoss.md): Computes the contrastive loss between `y_true` and `y_pred`.

[`class LiftedStructLoss`](../tfa/losses/LiftedStructLoss.md): Computes the lifted structured loss.

[`class NpairsLoss`](../tfa/losses/NpairsLoss.md): Computes the npairs loss between `y_true` and `y_pred`.

[`class NpairsMultilabelLoss`](../tfa/losses/NpairsMultilabelLoss.md): Computes the npairs loss between multilabel data `y_true` and `y_pred`.

[`class SigmoidFocalCrossEntropy`](../tfa/losses/SigmoidFocalCrossEntropy.md): Implements the focal loss function.

[`class SparsemaxLoss`](../tfa/losses/SparsemaxLoss.md): Sparsemax loss function.

[`class TripletSemiHardLoss`](../tfa/losses/TripletSemiHardLoss.md): Computes the triplet loss with semi-hard negative mining.

## Functions

[`contrastive_loss(...)`](../tfa/losses/contrastive_loss.md): Computes the contrastive loss between `y_true` and `y_pred`.

[`lifted_struct_loss(...)`](../tfa/losses/lifted_struct_loss.md): Computes the lifted structured loss.

[`npairs_loss(...)`](../tfa/losses/npairs_loss.md): Computes the npairs loss between `y_true` and `y_pred`.

[`npairs_multilabel_loss(...)`](../tfa/losses/npairs_multilabel_loss.md): Computes the npairs loss between multilabel data `y_true` and `y_pred`.

[`sigmoid_focal_crossentropy(...)`](../tfa/losses/sigmoid_focal_crossentropy.md): Args

[`sparsemax_loss(...)`](../tfa/losses/sparsemax_loss.md): Sparsemax loss function [1].

[`triplet_semihard_loss(...)`](../tfa/losses/triplet_semihard_loss.md): Computes the triplet loss with semi-hard negative mining.

