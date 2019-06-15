<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.losses" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfa.losses

Additional losses that conform to Keras API.



Defined in [`losses/__init__.py`](https://github.com/tensorflow/addons/tree/0.4-release/tensorflow_addons/losses/__init__.py).

<!-- Placeholder for "Used in" -->


## Modules

[`contrastive`](../tfa/losses/contrastive.md) module: Implements contrastive loss.

[`focal_loss`](../tfa/losses/focal_loss.md) module: Implements Focal loss.

[`lifted`](../tfa/losses/lifted.md) module: Implements lifted_struct_loss.

[`metric_learning`](../tfa/losses/metric_learning.md) module: Functions of metric learning.

[`triplet`](../tfa/losses/triplet.md) module: Implements triplet loss.

## Classes

[`class ContrastiveLoss`](../tfa/losses/ContrastiveLoss.md): Computes the contrastive loss between `y_true` and `y_pred`.

[`class LiftedStructLoss`](../tfa/losses/LiftedStructLoss.md): Computes the lifted structured loss.

[`class SigmoidFocalCrossEntropy`](../tfa/losses/SigmoidFocalCrossEntropy.md): Implements the focal loss function.

[`class SparsemaxLoss`](../tfa/losses/SparsemaxLoss.md): Sparsemax loss function.

[`class TripletSemiHardLoss`](../tfa/losses/TripletSemiHardLoss.md): Computes the triplet loss with semi-hard negative mining.

## Functions

[`contrastive_loss(...)`](../tfa/losses/contrastive_loss.md): Computes the contrastive loss between `y_true` and `y_pred`.

[`lifted_struct_loss(...)`](../tfa/losses/lifted_struct_loss.md): Computes the lifted structured loss.

[`sigmoid_focal_crossentropy(...)`](../tfa/losses/sigmoid_focal_crossentropy.md): Args

[`sparsemax_loss(...)`](../tfa/losses/sparsemax_loss.md): Sparsemax loss function [1].

[`triplet_semihard_loss(...)`](../tfa/losses/triplet_semihard_loss.md): Computes the triplet loss with semi-hard negative mining.

