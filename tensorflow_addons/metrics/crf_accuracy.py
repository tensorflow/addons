import tensorflow as tf

from tensorflow_addons.utils import keras_utils


def _get_accuracy(y_true, y_pred, mask, sparse_target=False):
    y_pred = tf.keras.backend.argmax(y_pred, -1)
    if sparse_target:
        y_true = tf.keras.backend.cast(y_true[:, :, 0], tf.keras.backend.dtype(y_pred))
    else:
        y_true = tf.keras.backend.argmax(y_true, -1)
    judge = tf.keras.backend.cast(tf.keras.backend.equal(y_pred, y_true), tf.keras.backend.floatx())
    if mask is None:
        return tf.keras.backend.mean(judge)
    else:
        mask = tf.keras.backend.cast(mask, tf.keras.backend.floatx())
        return tf.keras.backend.sum(judge * mask) / tf.keras.backend.sum(mask)


def crf_viterbi_accuracy(y_true, y_pred):
    """
    Use Viterbi algorithm to get best path, and compute its accuracy.
    `y_pred` must be an output from CRF.
    """
    crf, idx = y_pred._keras_history[:2]
    return crf.get_accuracy(y_true, y_pred)


def crf_marginal_accuracy(y_true, y_pred):
    """
    Use time-wise marginal argmax as prediction.
    `y_pred` must be an output from CRF with `learn_mode="marginal"`.
    """
    crf, idx = y_pred._keras_history[:2]
    X = crf._inbound_nodes[idx].input_tensors[0]
    mask = crf._inbound_nodes[idx].input_masks[0]
    y_pred = crf.get_marginal_prob(X, mask)
    return _get_accuracy(y_true, y_pred, mask, crf.sparse_target)


@keras_utils.register_keras_custom_object
def crf_accuracy(y_true, y_pred):
    # TODO: using tf 2.0 class based implementation
    """
    Ge default accuracy based on CRF `test_mode`.
    """
    crf, idx = y_pred._keras_history[:2]
    if crf.test_mode == 'viterbi':
        return crf_viterbi_accuracy(y_true, y_pred)
    else:
        return crf_marginal_accuracy(y_true, y_pred)