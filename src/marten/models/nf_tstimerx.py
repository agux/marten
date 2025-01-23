import os

os.environ["NIXTLA_ID_AS_COL"] = "True"

import logging
import pandas as pd
import numpy as np
import math
import torch
import psutil
import random

from types import SimpleNamespace
from typing import Any, Tuple

from dask.distributed import get_worker

from neuralforecast import NeuralForecast
from neuralforecast.models import TSMixerx
from neuralforecast.losses.pytorch import HuberLoss

from marten.models.base_model import BaseModel
from marten.utils.worker import num_workers
from marten.utils.logger import get_logger

# import zentorch

default_params = {
    "h": 20,
    "max_steps": 1000,
    "val_check_steps": 50,
    "num_lr_decays": -1,
    "early_stop_patience_steps": 10,
    "accelerator": "auto",
}

baseline_params = {
    "input_size": 60,
    "n_block": 2,
    "ff_dim": 16,
    "dropout": 0.1,
    "revin": True,
    "learning_rate": 1e-3,
    "batch_size": 256,
    # step_size = 1,
    # "random_seed": 7,
    "optimizer": "AdamW",
    # num_workers_loader = 0,
    # drop_last_loader = False,
    # lr_scheduler=None,
    # lr_scheduler_kwargs=None,
    # NOTE: scaling will be set when instantiating `NeuralForecast`.
    # See https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/time_series_scaling.html
    # scaler_type="identity",
    "local_scaler_type": None,  # Can be ‘standard’, ‘robust’, ‘robust-iqr’, ‘minmax’ or ‘boxcox’ (positive variables only)
}


class TSMixerxModel(BaseModel):

    def __init__(self) -> None:
        super(TSMixerxModel, self).__init__()
        self.model = None
        self.nf = None
        self.val_size = None

        # torch.set_num_interop_threads(1)

    def restore_params(self, params: dict, **kwargs: Any) -> dict:
        return params

    def power_demand(self, args: SimpleNamespace, params: dict) -> int:
        return 2

    def gpu_threshold(self) -> Tuple[float, float]:
        if self.is_baseline(**self.model_args):
            return 40, 50
        else:
            return 35, 35

    def cpu_threshold(self) -> Tuple[float, float]:
        if self.is_baseline(**self.model_args):
            return 60, 70
        else:
            return 40, 60

    def trainable_on_cpu(self, **kwargs: Any) -> bool:
        if "num_covars" not in kwargs:
            return True
        return (
            math.pow(
                (
                    kwargs["ff_dim"]
                    * kwargs["n_block"]
                    * kwargs["batch_size"]
                    * kwargs["input_size"]
                    * kwargs["num_covars"]
                ),
                0.2,
            )
            < 70
        )
        # return True

    def torch_num_threads(self) -> float:
        is_baseline = self.is_baseline(**self.model_args)
        if is_baseline:
            return int(os.getenv("tsmixerx_torch_num_threads", 1))
        else:
            n_workers = num_workers()
            cpu_count = psutil.cpu_count(logical=True)
            quotient = math.ceil(cpu_count / n_workers)
            choices = [n for n in range(2, quotient + 3)]
            return random.choice(choices)

    def _train(self, df: pd.DataFrame, **kwargs: Any) -> dict:
        model_config = default_params.copy()
        model_config.update(kwargs)
        # prep the parameters and df
        optimizer, optim_args = self._select_optimizer(**model_config)

        seed_logger = logging.getLogger("lightning_fabric.utilities.seed")
        from lightning_utilities.core.rank_zero import log as rank_zero_logger

        # rank_zero_logger = logging.getLogger("lightning_utilities.core.rank_zero")
        orig_seed_log_level = seed_logger.getEffectiveLevel()
        orig_log_level = rank_zero_logger.getEffectiveLevel()
        seed_logger.setLevel(logging.FATAL)
        rank_zero_logger.setLevel(logging.FATAL)

        exog = [col for col in df.columns if col not in ["unique_id", "ds", "y"]]

        self.model = TSMixerx(
            h=model_config["h"],
            n_series=1,
            input_size=model_config["input_size"],
            # stat_exog_list=None,
            hist_exog_list=exog,
            # futr_exog_list=None,
            n_block=model_config["n_block"],
            ff_dim=model_config["ff_dim"],
            dropout=model_config["dropout"],
            revin=model_config["revin"],
            loss=HuberLoss(),
            # valid_loss=None,
            max_steps=model_config["max_steps"],
            learning_rate=model_config["learning_rate"],
            num_lr_decays=model_config["num_lr_decays"],
            early_stop_patience_steps=(
                model_config["early_stop_patience_steps"]
                if model_config["validate"]
                else -1
            ),
            val_check_steps=model_config["val_check_steps"],
            batch_size=model_config["batch_size"],
            # step_size = 1,
            random_seed=model_config["random_seed"],
            # num_workers_loader = 0,
            # drop_last_loader = False,
            optimizer=optimizer,
            optimizer_kwargs=optim_args,
            # lr_scheduler=None,
            # lr_scheduler_model_config=None,
            # NOTE: scaling will be set when instantiating `NeuralForecast`.
            # See https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/time_series_scaling.html
            # scaler_type="identity",
            # NOTE: beginning of trainer_kwargs
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator=model_config["accelerator"],
            # devices="auto",  #NOTE: not workable for CPU
            devices=model_config["devices"],
            # precision="bf16-mixed",  #NOTE: saves GPU mem but slower on CPU
            enable_checkpointing=False,
            logger=self.csvLogger,  # NOTE: can't disable logger as early stopping rely on it
            # barebones=True, # NOTE: this disable logger as well
        )

        if (
            model_config["accelerator"] == "cpu"
            and self.zentorch_enabled
            and not self.is_baseline(**self.model_args)
        ):
            get_logger().info(
                "Enabling Zentorch. accelerator: %s, zentorch_enabled: %s",
                model_config["accelerator"],
                self.zentorch_enabled,
            )
            # NOTE: many inductor sub-process will be spawn if zentorch is imported
            import zentorch

            self.model = torch.compile(self.model, backend="zentorch")

        self.nf = NeuralForecast(
            models=[self.model],
            freq="B",
            # Scaler to apply per-serie to all features before fitting, which is inverted after predicting.
            # Can be 'standard', 'robust', 'robust-iqr', 'minmax' or 'boxcox'
            local_scaler_type=model_config["local_scaler_type"],
        )
        self.val_size = min(300, int(len(df) * 0.9)) if model_config["validate"] else 0
        self.nf.fit(
            df,
            val_size=self.val_size,
            # use_init_models=True
        )

        seed_logger.setLevel(orig_seed_log_level)
        rank_zero_logger.setLevel(orig_log_level)

        return model_config

    def baseline_params(self) -> dict:
        return baseline_params.copy()

    def trim_forecast(self, forecast: pd.DataFrame) -> pd.DataFrame:
        return forecast[["ds", "yhat_n"]].copy()

    def _predict(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        forecast = self.nf.predict(df)
        # check if "id" column is in the forecast dataframe. If so, drop this column.
        if "index" in forecast.columns:
            forecast.drop(columns=["index"], inplace=True)
        forecast.reset_index(drop=True, inplace=True)
        forecast.insert(forecast.columns.get_loc("ds") + 1, "y", np.nan)
        forecast.rename(columns={"TSMixerx": "yhat_n"}, inplace=True)
        return forecast

    def search_space(self, **kwargs: Any) -> str:
        # TODO: add random_seed to search space?
        # "boxcox" local_scaler_type supports positive variables only
        return f"""dict(
            input_size=range(5, 500+1),
            n_block=range(2, 256+1),
            ff_dim=range(2, 256+1),
            dropout=uniform(0, 0.5),
            revin=[True, False],
            learning_rate=loguniform(0.0001, 0.002),
            batch_size=[2**w for w in range(5, 7+1)],
            local_scaler_type=[None, "standard", "robust", "robust-iqr", "minmax"],
            topk_covar=list(range(0, {kwargs["topk_covars"]}+1)),
            covar_dist=dirichlet([float({kwargs["max_covars"]})]*{kwargs["max_covars"]}),
            optimizer=["Adam", "AdamW", "SGD"],
        )"""

    def accept_missing_data(self) -> bool:
        return False
