<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.seq2seq.sampler" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfa.seq2seq.sampler

A library of sampler for use with SamplingDecoders.



Defined in [`seq2seq/sampler.py`](https://github.com/tensorflow/addons/tree/0.4-release/tensorflow_addons/seq2seq/sampler.py).

<!-- Placeholder for "Used in" -->


## Classes

[`class CustomSampler`](../../tfa/seq2seq/CustomSampler.md): Base abstract class that allows the user to customize sampling.

[`class GreedyEmbeddingSampler`](../../tfa/seq2seq/GreedyEmbeddingSampler.md): A sampler for use during inference.

[`class InferenceSampler`](../../tfa/seq2seq/InferenceSampler.md): A helper to use during inference with a custom sampling function.

[`class SampleEmbeddingSampler`](../../tfa/seq2seq/SampleEmbeddingSampler.md): A sampler for use during inference.

[`class Sampler`](../../tfa/seq2seq/Sampler.md): Interface for implementing sampling in seq2seq decoders.

[`class ScheduledEmbeddingTrainingSampler`](../../tfa/seq2seq/ScheduledEmbeddingTrainingSampler.md): A training sampler that adds scheduled sampling.

[`class ScheduledOutputTrainingSampler`](../../tfa/seq2seq/ScheduledOutputTrainingSampler.md): A training sampler that adds scheduled sampling directly to outputs.

[`class TrainingSampler`](../../tfa/seq2seq/TrainingSampler.md): A Sampler for use during training.

## Functions

[`bernoulli_sample(...)`](../../tfa/seq2seq/sampler/bernoulli_sample.md): Samples from Bernoulli distribution.

[`categorical_sample(...)`](../../tfa/seq2seq/sampler/categorical_sample.md): Samples from categorical distribution.

