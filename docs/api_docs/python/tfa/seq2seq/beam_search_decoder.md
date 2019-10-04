<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.seq2seq.beam_search_decoder" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfa.seq2seq.beam_search_decoder


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.6/tensorflow_addons/seq2seq/beam_search_decoder.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



A decoder that performs beam search.

<!-- Placeholder for "Used in" -->


## Classes

[`class BeamSearchDecoder`](../../tfa/seq2seq/BeamSearchDecoder.md): BeamSearch sampling decoder.

[`class BeamSearchDecoderMixin`](../../tfa/seq2seq/beam_search_decoder/BeamSearchDecoderMixin.md): BeamSearchDecoderMixin contains the common methods for

[`class BeamSearchDecoderOutput`](../../tfa/seq2seq/BeamSearchDecoderOutput.md)

[`class BeamSearchDecoderState`](../../tfa/seq2seq/BeamSearchDecoderState.md)

[`class FinalBeamSearchDecoderOutput`](../../tfa/seq2seq/FinalBeamSearchDecoderOutput.md): Final outputs returned by the beam search after all decoding is

## Functions

[`attention_probs_from_attn_state(...)`](../../tfa/seq2seq/beam_search_decoder/attention_probs_from_attn_state.md): Calculates the average attention probabilities.

[`gather_tree_from_array(...)`](../../tfa/seq2seq/gather_tree_from_array.md): Calculates the full beams for `TensorArray`s.

[`get_attention_probs(...)`](../../tfa/seq2seq/beam_search_decoder/get_attention_probs.md): Get attention probabilities from the cell state.

[`tile_batch(...)`](../../tfa/seq2seq/tile_batch.md): Tile the batch dimension of a (possibly nested structure of) tensor(s)

