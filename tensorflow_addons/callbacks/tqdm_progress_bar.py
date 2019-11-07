# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""TQDM Progress Bar."""

from __future__ import absolute_import, division, print_function

import time
import tensorflow as tf
from collections import defaultdict

from tensorflow.keras.callbacks import Callback


@tf.keras.utils.register_keras_serializable(package='Addons')
class TQDMProgressBar(Callback):
    """TQDM Progress Bar for Tensorflow Keras.

    Arguments:
        metrics_separator (string): Custom separator between metrics.
            Defaults to ' - '
        overall_bar_format (string format): Custom bar format for overall
            (outer) progress bar, see https://github.com/tqdm/tqdm#parameters
            for more detail.
        epoch_bar_format (string format): Custom bar format for epoch
            (inner) progress bar, see https://github.com/tqdm/tqdm#parameters
            for more detail.
        update_per_second (int): Maximum number of updates in the epochs bar
            per second, this is to prevent small batches from slowing down
            training. Defaults to 10.
        leave_epoch_progress (bool): True to leave epoch progress bars
        leave_overall_progress (bool): True to leave overall progress bar
        show_epoch_progress (bool): False to hide epoch progress bars
        show_overall_progress (bool): False to hide overall progress bar
    """

    def __init__(self,
                 metrics_separator=" - ",
                 overall_bar_format='{l_bar}{bar} {n_fmt}/{total_fmt} ETA: '
                 '{remaining}s,  {rate_fmt}{postfix}',
                 epoch_bar_format='{n_fmt}/{total_fmt}{bar} ETA: '
                 '{remaining}s - {desc}',
                 update_per_second=10,
                 leave_epoch_progress=True,
                 leave_overall_progress=True,
                 show_epoch_progress=True,
                 show_overall_progress=True):

        try:
            # import tqdm here because tqdm is not a required package
            # for addons
            import tqdm
            version_message = 'Please update your TQDM version to >= 4.36.1, '
            'you have version {}. To update, run !pip install -U tqdm'
            assert tqdm.__version__ >= '4.36.1', version_message.format(
                tqdm.__version__)
            from tqdm.auto import tqdm
            self.tqdm = tqdm
        except ImportError:
            raise ImportError("Please install tqdm via pip install tqdm")

        self.metrics_separator = metrics_separator
        self.overall_bar_format = overall_bar_format
        self.epoch_bar_format = epoch_bar_format
        self.leave_epoch_progress = leave_epoch_progress
        self.leave_overall_progress = leave_overall_progress
        self.show_epoch_progress = show_epoch_progress
        self.show_overall_progress = show_overall_progress

        # compute update interval (inverse of update per second)
        self.update_interval = 1 / update_per_second

        self.last_update_time = time.time()
        self.overall_progress_tqdm = None
        self.epoch_progress_tqdm = None
        self.num_epochs = None
        self.logs = None
        self.metrics = None

    def on_train_begin(self, logs=None):
        self.num_epochs = self.params['epochs']
        self.metrics = self.params['metrics']

        if self.show_overall_progress:
            self.overall_progress_tqdm = self.tqdm(
                desc='Training',
                total=self.num_epochs,
                bar_format=self.overall_bar_format,
                leave=self.leave_overall_progress,
                dynamic_ncols=True,
                unit='epochs')

        # set counting mode
        if 'samples' in self.params:
            self.mode = 'samples'
            self.total_steps = self.params['samples']
        else:
            self.mode = 'steps'
            self.total_steps = self.params['steps']

    def on_train_end(self, logs={}):
        if self.show_overall_progress:
            self.overall_progress_tqdm.close()

    def on_epoch_begin(self, epoch, logs={}):
        current_epoch_description = "Epoch {epoch}/{num_epochs}".format(
            epoch=epoch + 1, num_epochs=self.num_epochs)

        if self.show_epoch_progress:
            print(current_epoch_description)
            self.epoch_progress_tqdm = self.tqdm(
                total=self.total_steps,
                bar_format=self.epoch_bar_format,
                leave=self.leave_epoch_progress,
                dynamic_ncols=True,
                unit=self.mode)

        self.seen = 0
        self.steps_to_update = 0
        self.logs = defaultdict(float)

    def on_epoch_end(self, epoch, logs={}):

        if self.show_epoch_progress:
            metrics = self.format_metrics(logs)
            self.epoch_progress_tqdm.desc = metrics

            # set miniters and mininterval to 0 so last update displays
            self.epoch_progress_tqdm.miniters = 0
            self.epoch_progress_tqdm.mininterval = 0

            # update the rest of the steps in epoch progress bar
            self.epoch_progress_tqdm.update(self.total_steps -
                                            self.epoch_progress_tqdm.n)
            self.epoch_progress_tqdm.close()

        if self.show_overall_progress:
            self.overall_progress_tqdm.update(1)

    def on_batch_end(self, batch, logs={}):
        if self.mode == "samples":
            batch_size = logs['size']
        else:
            batch_size = 1

        self.seen += batch_size
        self.steps_to_update += batch_size

        if self.seen < self.total_steps:

            for metric, value in logs.items():
                self.logs[metric] += value * batch_size

            now = time.time()
            time_diff = now - self.last_update_time
            if self.show_epoch_progress and time_diff >= self.update_interval:

                # update the epoch progress bar
                metrics = self.format_metrics(self.logs, self.seen)
                self.epoch_progress_tqdm.desc = metrics
                self.epoch_progress_tqdm.update(self.steps_to_update)

                # reset steps to update
                self.steps_to_update = 0

                # update timestamp for last update
                self.last_update_time = now

    def format_metrics(self, logs={}, factor=1):
        """Format metrics in logs into a string.

        Arguments:
            logs: dictionary of metrics and their values. Defaults to
                empty dictionary.
            factor (int): The factor we want to divide the metrics in logs
                by, useful when we are computing the logs after each batch.
                Defaults to 1.

        Returns:
            metrics_string: a string displaying metrics using the given
            formators passed in through the constructor.
        """

        metric_value_pairs = []
        for metric in self.metrics:
            if metric in logs:
                value = logs[metric] / factor
                pair = '{name}: {value:0.4f}'.format(name=metric, value=value)
                metric_value_pairs.append(pair)
        metrics_string = self.metrics_separator.join(metric_value_pairs)
        return metrics_string

    def get_config(self):
        config = {
            'metrics_separator': self.metrics_separator,
            'overall_bar_format': self.overall_bar_format,
            'epoch_bar_format': self.epoch_bar_format,
            'leave_epoch_progress': self.leave_epoch_progress,
            'leave_overall_progress': self.leave_overall_progress,
            'show_epoch_progress': self.show_epoch_progress,
            'show_overall_progress': self.show_overall_progress,
        }

        base_config = super(TQDMProgressBar, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
