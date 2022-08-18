# Copyright (c) 2022, NVIDIA CORPORATION.
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

import dfencoder
import numpy as np


class FixedStandardScalar(dfencoder.StandardScaler):

    def fit(self, x):
        super().fit(x)

        # Having a std == 0 (when all values are the same), breaks training. Just use 1.0 in this case
        if (self.std == 0.0):
            self.std = 1.0


class DFPAutoEncoder(dfencoder.AutoEncoder):

    def __init__(self,
                 encoder_layers=None,
                 decoder_layers=None,
                 encoder_dropout=None,
                 decoder_dropout=None,
                 encoder_activations=None,
                 decoder_activations=None,
                 activation='relu',
                 min_cats=10,
                 swap_p=0.15,
                 lr=0.01,
                 batch_size=256,
                 eval_batch_size=1024,
                 optimizer='adam',
                 amsgrad=False,
                 momentum=0,
                 betas=...,
                 dampening=0,
                 weight_decay=0,
                 lr_decay=None,
                 nesterov=False,
                 verbose=False,
                 device=None,
                 logger='basic',
                 logdir='logdir/',
                 project_embeddings=True,
                 run=None,
                 progress_bar=True,
                 n_megabatches=1,
                 scaler='standard',
                 *args,
                 **kwargs):
        super().__init__(encoder_layers,
                         decoder_layers,
                         encoder_dropout,
                         decoder_dropout,
                         encoder_activations,
                         decoder_activations,
                         activation,
                         min_cats,
                         swap_p,
                         lr,
                         batch_size,
                         eval_batch_size,
                         optimizer,
                         amsgrad,
                         momentum,
                         betas,
                         dampening,
                         weight_decay,
                         lr_decay,
                         nesterov,
                         verbose,
                         device,
                         logger,
                         logdir,
                         project_embeddings,
                         run,
                         progress_bar,
                         n_megabatches,
                         scaler,
                         *args,
                         **kwargs)

        self.val_loss_mean = None
        self.val_loss_std = None

    def get_scaler(self, name):
        scaler_result = super().get_scaler(name)

        # Use the fixed scalar instead of the standard
        if (scaler_result == dfencoder.StandardScaler):
            return FixedStandardScalar

        return scaler_result

    def fit(self, df, epochs=1, val=None):
        super().fit(df, epochs, val)

        # Before returning, calc quick validation stats
        if (val is not None):

            val_loss = self.get_anomaly_score(val)

            self.val_loss_mean = np.mean(val_loss)
            self.val_loss_std = np.std(val_loss)
