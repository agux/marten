import pandas as pd
import numpy as np
import logging
import torch
from neuralforecast.core import NeuralForecast
from neuralforecast.models import TSMixerx, NHITS

# Prep dummy data
start_date = "2024-06-01"
end_date = "2025-02-19"
date_range = pd.date_range(start=start_date, end=end_date, freq="B")
np.random.seed(0)
df = pd.DataFrame(
    {
        "unique_id": "dummy",
        "ds": date_range,
        "y": np.random.randn(len(date_range)),
        "val1": np.random.randn(len(date_range)),
        "val2": np.random.rand(len(date_range)),
    }
)

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
torch.set_float32_matmul_precision("medium")

horizon = 10
input_size = 30
val_size = 50

def tsmixerx():
    models = [
        TSMixerx(
            h=horizon,
            input_size=input_size,
            n_series=1,
            max_steps=100,
            random_seed=0,
            hist_exog_list=["val1", "val2"],
        ),
    ]

    nf = NeuralForecast(
        models=models,
        freq="B",
        local_scaler_type="robust",
    )
    nf.fit(df=df, val_size=val_size)

    Y_hat_insample = nf.predict_insample(step_size=horizon)

def nhits():
    models = [
        NHITS(
            h=horizon,  # Forecast horizon
            input_size=input_size,  # Length of input sequence
            max_steps=100,  # Number of steps to train
            n_freq_downsample=[2, 1, 1],  # Downsampling factors for each stack output
            mlp_units=3 * [[1024, 1024]],
        )  # Number of units in each block.
    ]
    nf = NeuralForecast(models=models, freq="B")
    nf.fit(df=df, val_size=val_size)

    Y_hat_insample = nf.predict_insample(step_size=horizon)


if __name__ == "__main__":
    # nhits()
    tsmixerx()