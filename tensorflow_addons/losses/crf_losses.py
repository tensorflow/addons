from tensorflow.python.keras.losses import sparse_categorical_crossentropy, \
    categorical_crossentropy

from tensorflow_addons.utils import keras_utils


def crf_nll(y_true, y_pred):
    crf, idx = y_pred._keras_history[:2]

    node = crf._inbound_nodes[idx]

    nloglik = crf.get_negative_log_likelihood(y_true)

    return nloglik


@keras_utils.register_keras_custom_object
def crf_loss(y_true, y_pred):
    # TODO: change to tf 2.0 class based implementation
    crf, idx = y_pred._keras_history[:2]

    if crf.learn_mode == 'join':
        return crf_nll(y_true, y_pred)
    else:
        if crf.sparse_target:
            return sparse_categorical_crossentropy(y_true, y_pred)
        else:
            return categorical_crossentropy(y_true, y_pred)
