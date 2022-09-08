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
import torch


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

    def get_results(self, df, return_abs=False):
        pdf = df.copy()
        self.eval()
        data = self.prepare_df(df)
        orig_cols = data.columns
        with torch.no_grad():
            num, bin, embeddings = self.encode_input(data)
            x = torch.cat(num + bin + embeddings, dim=1)
            x = self.encode(x)
            output_df = self.decode_to_df(x, df=df)
        mse, bce, cce, _ = self.get_anomaly_score(df)
        mse_scaled, bce_scaled, cce_scaled = self.get_scaled_anomaly_scores(df)
        for i, ft in enumerate(self.numeric_fts):
            pdf[ft + '_pred'] = output_df[ft]
            pdf[ft + '_loss'] = mse[:, i].cpu().numpy()
            pdf[ft + '_z_loss'] = mse_scaled[:, i].cpu().numpy() if not return_abs else abs(mse_scaled[:,
                                                                                                       i].cpu().numpy())
        for i, ft in enumerate(self.binary_fts):
            pdf[ft + '_pred'] = output_df[ft]
            pdf[ft + '_loss'] = bce[:, i].cpu().numpy()
            pdf[ft + '_z_loss'] = bce_scaled[:, i].cpu().numpy() if not return_abs else abs(bce_scaled[:,
                                                                                                       i].cpu().numpy())
        for i, ft in enumerate(self.categorical_fts):
            pdf[ft + '_pred'] = output_df[ft]
            pdf[ft + '_loss'] = cce[i].cpu().numpy()
            pdf[ft + '_z_loss'] = cce_scaled[i].cpu().numpy() if not return_abs else abs(cce_scaled[i].cpu().numpy())
        all_cols = [[c, c + '_pred', c + '_loss', c + '_z_loss'] for c in orig_cols]
        result_cols = [col for col_collection in all_cols for col in col_collection]
        z_losses = [c + '_z_loss' for c in orig_cols]
        pdf['max_abs_z'] = pdf[z_losses].max(axis=1)
        pdf['mean_abs_z'] = pdf[z_losses].mean(axis=1)
        result_cols.append('max_abs_z')
        result_cols.append('mean_abs_z')
        return pdf[result_cols]
