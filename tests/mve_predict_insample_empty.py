import pandas as pd
import numpy as np
import logging
import torch
from neuralforecast.core import NeuralForecast
from neuralforecast.models import TSMixerx, NHITS
from neuralforecast.losses.pytorch import HuberLoss

# Prep data
df_train = pd.read_pickle("input_df_20250405184630794.pkl")
exog = [col for col in df_train.columns if col not in ["unique_id", "ds", "y"]]

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
torch.set_float32_matmul_precision("medium")

val_size = 0

def tsmixerx():
    horizon = 20
    import torch.optim as optim
    model = TSMixerx(
        h=horizon,
        input_size=60,
        n_series=1,
        dropout=0.1,
        # stat_exog_list=["airline1"],
        # futr_exog_list=exog,
        hist_exog_list=exog,
        n_block=2,
        ff_dim=16,
        revin=True,
        # scaler_type="standard",
        max_steps=500,
        early_stop_patience_steps=-1,
        val_check_steps=5,
        learning_rate=1e-3,
        # enable_lr_find=True,
        loss=HuberLoss(),
        valid_loss=HuberLoss(),
        batch_size=32,
        random_seed=7,
        optimizer=optim.AdamW,
        optimizer_kwargs={"fused": False},
    )
    nf = NeuralForecast(
        models=[model],
        freq="B",
        local_scaler_type=None,
    )
    nf.fit(df=df_train, val_size=val_size)

    Y_hat_insample = nf.predict_insample(step_size=horizon)

    Y_hat_insample.info()

if __name__ == "__main__":
    # nhits()
    tsmixerx()
