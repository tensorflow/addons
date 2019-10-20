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
"""TQDM Progress Bar. """

from __future__ import absolute_import, division, print_function

from sys import stderr

import numpy as np
import six
from tensorflow.keras.callbacks import Callback
from tensorflow_addons.utils import keras_utils
from tqdm.auto import tqdm


@keras_utils.register_keras_custom_object
class TQDMProgressBar(Callback):
    """TQDM Progress Bar for Tensorflow Keras.

    Arguments:
        outer_description: string for outer progress bar
        inner_description_initial: initial format for epoch ("Epoch: {epoch}")
        inner_description_update: format after metrics collected ("Epoch: {epoch} - {metrics}")
        metric_format: format for each metric name/value pair ("{name}: {value:0.3f}")
        separator: separator between metrics (" - ")
        leave_inner: True to leave inner bars
        leave_outer: True to leave outer bars
        show_inner: False to hide inner bars
        show_outer: False to hide outer bar
    """

    def __init__(self, outer_description="Training",
                 inner_description_initial="Epoch {epoch}/{num_epochs}",
                 inner_description_update="{metrics}",
                 metric_format="{name}: {value:0.4f}",
                 separator=" - ",
                 leave_inner=True,
                 leave_outer=True,
                 show_inner=True,
                 show_outer=True):

        self.outer_description = outer_description
        self.inner_description_initial = inner_description_initial
        self.inner_description_update = inner_description_update
        self.metric_format = metric_format
        self.separator = separator
        self.leave_inner = leave_inner
        self.leave_outer = leave_outer
        self.show_inner = show_inner
        self.show_outer = show_outer

        self.tqdm_outer = None
        self.tqdm_inner = None
        self.epoch = None
        self.num_epochs = None
        self.running_logs = None
        self.inner_count = None

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch = epoch
        desc = self.inner_description_initial.format(epoch=self.epoch + 1, num_epochs=self.num_epochs)
        if self.mode == 'samples':
            self.inner_total = self.params['samples']
        else:
            self.inner_total = self.params['steps']
        if self.show_inner:
            print(desc)
            self.tqdm_inner = tqdm(
                total=self.inner_total, leave=self.leave_inner, dynamic_ncols=True, unit=self.mode, bar_format='{r_bar}{bar}{l_bar}')
        self.inner_count = 0
        self.running_logs = {}

    def on_epoch_end(self, epoch, logs={}):
        metrics = self.format_metrics(logs)
        # desc = self.inner_description_update.format(metrics=metrics)
        desc = metrics
        if self.show_inner:
            self.tqdm_inner.desc = desc
            # set miniters and mininterval to 0 so last update displays
            self.tqdm_inner.miniters = 0
            self.tqdm_inner.mininterval = 0
            self.tqdm_inner.update(self.inner_total - self.tqdm_inner.n)
            self.tqdm_inner.close()
        if self.show_outer:
            self.tqdm_outer.update(1)

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        if self.mode == "samples":
            update = logs['size']
        else:
            update = 1
        self.inner_count += update
        if self.inner_count < self.inner_total:
            self.append_logs(logs)
            metrics = self.format_metrics(self.running_logs)
            # desc = self.inner_description_update.format(
            #   epoch=self.epoch, metrics=metrics)
            desc = metrics
            if self.show_inner:
                self.tqdm_inner.desc = desc
                self.tqdm_inner.update(update)

    def on_train_begin(self, logs={}):
        self.num_epochs = self.params['epochs']
        if self.show_outer:
            self.tqdm_outer = tqdm(
                desc=self.outer_description, total=self.num_epochs, leave=self.leave_outer, dynamic_ncols=True, unit="epochs")

        # set counting mode
        if 'samples' in self.params:
            self.mode = 'samples'
        else:
            self.mode = 'steps'

    def on_train_end(self, logs={}):
        if self.show_outer:
            self.tqdm_outer.close()

    def append_logs(self, logs):
        """append logs seen in a batch to the running log to display updated 
            metrics values in real time."""
        metrics = self.params['metrics']
        for metric, value in six.iteritems(logs):
            if metric in metrics:
                if metric in self.running_logs:
                    self.running_logs[metric].append(value[()])
                else:
                    self.running_logs[metric] = [value[()]]

    def format_metrics(self, logs):
        """Format metrics in logs into a string.

        Arguments:
            logs: dictionary of metrics and their values.

        Returns:
            metrics_string: a string displaying metrics using the given 
            formators passed in through the constructor.
        """
        metrics = self.params['metrics']
        metric_value_pairs = []
        for metric in metrics:
            if metric in logs:
                pair = self.metric_format.format(
                    name=metric, value=np.mean(logs[metric], axis=None))
                metric_value_pairs.append(pair)
        metrics_string = self.separator.join(metric_value_pairs)
        return metrics_string
