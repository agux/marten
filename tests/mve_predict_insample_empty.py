import pandas as pd
import numpy as np
import logging
import torch
from neuralforecast.core import NeuralForecast
from neuralforecast.models import TSMixerx, NHITS
from neuralforecast.losses.pytorch import HuberLoss
import time

torch.set_num_threads(1)

# Prep data
df_train = pd.read_pickle("input_df.pkl")
print(df_train)

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
        # dropout=0.1,
        # stat_exog_list=["airline1"],
        # futr_exog_list=exog,
        hist_exog_list=exog,
        # num_lr_decays=-1,
        # n_block=2,
        # ff_dim=32,
        # revin=True,
        # scaler_type="standard",
        # max_steps=10000,
        early_stop_patience_steps=-1,
        val_check_steps=50,
        learning_rate=1e-3,
        # enable_lr_find=True,
        # loss=HuberLoss(),
        batch_size=32,
        random_seed=7,
        # optimizer=optim.AdamW,
        # optimizer_kwargs={"fused": False},
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        log_every_n_steps=10,
    )
    nf = NeuralForecast(
        models=[model],
        freq="B",
        local_scaler_type=None,
    )
    nf.fit(df=df_train, val_size=val_size)

    Y_hat_insample = nf.predict_insample(step_size=horizon)

    Y_hat_insample.info()
    print(nf.models[0].hparams)


if __name__ == "__main__":
    start = time.perf_counter()

    # nhits()
    tsmixerx()

    end = time.perf_counter()
    print(f"Elapsed time: {end - start} seconds")
