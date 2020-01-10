<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.seq2seq" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfa.seq2seq


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/seq2seq/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Additional ops for building neural network sequence to sequence decoders and

losses.

## Modules

[`attention_wrapper`](../tfa/seq2seq/attention_wrapper.md) module: A powerful dynamic attention wrapper object.

[`basic_decoder`](../tfa/seq2seq/basic_decoder.md) module: A class of Decoders that may sample to generate the next input.

[`beam_search_decoder`](../tfa/seq2seq/beam_search_decoder.md) module: A decoder that performs beam search.

[`decoder`](../tfa/seq2seq/decoder.md) module: Seq2seq layer operations for use in neural networks.

[`loss`](../tfa/seq2seq/loss.md) module: Seq2seq loss operations for use in sequence models.

[`sampler`](../tfa/seq2seq/sampler.md) module: A library of sampler for use with SamplingDecoders.

## Classes

[`class AttentionMechanism`](../tfa/seq2seq/AttentionMechanism.md)

[`class AttentionWrapper`](../tfa/seq2seq/AttentionWrapper.md): Wraps another `RNNCell` with attention.

[`class AttentionWrapperState`](../tfa/seq2seq/AttentionWrapperState.md): `namedtuple` storing the state of a `AttentionWrapper`.

[`class BahdanauAttention`](../tfa/seq2seq/BahdanauAttention.md): Implements Bahdanau-style (additive) attention.

[`class BahdanauMonotonicAttention`](../tfa/seq2seq/BahdanauMonotonicAttention.md): Monotonic attention mechanism with Bahadanau-style energy function.

[`class BaseDecoder`](../tfa/seq2seq/BaseDecoder.md): An RNN Decoder that is based on a Keras layer.

[`class BasicDecoder`](../tfa/seq2seq/BasicDecoder.md): Basic sampling decoder.

[`class BasicDecoderOutput`](../tfa/seq2seq/BasicDecoderOutput.md): BasicDecoderOutput(rnn_output, sample_id)

[`class BeamSearchDecoder`](../tfa/seq2seq/BeamSearchDecoder.md): BeamSearch sampling decoder.

[`class BeamSearchDecoderOutput`](../tfa/seq2seq/BeamSearchDecoderOutput.md): BeamSearchDecoderOutput(scores, predicted_ids, parent_ids)

[`class BeamSearchDecoderState`](../tfa/seq2seq/BeamSearchDecoderState.md): BeamSearchDecoderState(cell_state, log_probs, finished, lengths, accumulated_attention_probs)

[`class CustomSampler`](../tfa/seq2seq/CustomSampler.md): Base abstract class that allows the user to customize sampling.

[`class Decoder`](../tfa/seq2seq/Decoder.md): An RNN Decoder abstract interface object.

[`class FinalBeamSearchDecoderOutput`](../tfa/seq2seq/FinalBeamSearchDecoderOutput.md): Final outputs returned by the beam search after all decoding is

[`class GreedyEmbeddingSampler`](../tfa/seq2seq/GreedyEmbeddingSampler.md): A sampler for use during inference.

[`class InferenceSampler`](../tfa/seq2seq/InferenceSampler.md): A helper to use during inference with a custom sampling function.

[`class LuongAttention`](../tfa/seq2seq/LuongAttention.md): Implements Luong-style (multiplicative) attention scoring.

[`class LuongMonotonicAttention`](../tfa/seq2seq/LuongMonotonicAttention.md): Monotonic attention mechanism with Luong-style energy function.

[`class SampleEmbeddingSampler`](../tfa/seq2seq/SampleEmbeddingSampler.md): A sampler for use during inference.

[`class Sampler`](../tfa/seq2seq/Sampler.md): Interface for implementing sampling in seq2seq decoders.

[`class ScheduledEmbeddingTrainingSampler`](../tfa/seq2seq/ScheduledEmbeddingTrainingSampler.md): A training sampler that adds scheduled sampling.

[`class ScheduledOutputTrainingSampler`](../tfa/seq2seq/ScheduledOutputTrainingSampler.md): A training sampler that adds scheduled sampling directly to outputs.

[`class SequenceLoss`](../tfa/seq2seq/SequenceLoss.md): Weighted cross-entropy loss for a sequence of logits.

[`class TrainingSampler`](../tfa/seq2seq/TrainingSampler.md): A Sampler for use during training.

## Functions

[`dynamic_decode(...)`](../tfa/seq2seq/dynamic_decode.md): Perform dynamic decoding with `decoder`.

[`gather_tree_from_array(...)`](../tfa/seq2seq/gather_tree_from_array.md): Calculates the full beams for `TensorArray`s.

[`hardmax(...)`](../tfa/seq2seq/hardmax.md): Returns batched one-hot vectors.

[`monotonic_attention(...)`](../tfa/seq2seq/monotonic_attention.md): Compute monotonic attention distribution from choosing probabilities.

[`safe_cumprod(...)`](../tfa/seq2seq/safe_cumprod.md): Computes cumprod of x in logspace using cumsum to avoid underflow.

[`sequence_loss(...)`](../tfa/seq2seq/sequence_loss.md): Weighted cross-entropy loss for a sequence of logits.

[`tile_batch(...)`](../tfa/seq2seq/tile_batch.md): Tile the batch dimension of a (possibly nested structure of) tensor(s)



