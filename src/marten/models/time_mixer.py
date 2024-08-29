import logging
import pandas as pd
import numpy as np

import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer

from types import SimpleNamespace
from typing import Any, Tuple, Type

from neuralforecast import NeuralForecast
from neuralforecast.models import TimeMixer
from neuralforecast.losses.pytorch import HuberLoss

from utilsforecast.losses import mae, rmse
from utilsforecast.evaluation import evaluate

from marten.models.base_model import BaseModel

default_params = {
    "h": 20,
    "max_steps": 1000,
    "val_check_steps": 50,
    "accelerator": "auto",
}

baseline_params = {
    "input_size": 60,
    "d_model": 32,
    "d_ff": 32,
    "dropout": 0.1,
    "e_layers": 4,
    "top_k": 5,
    "decomp_method": "moving_avg",  # moving_avg, dft_decomp
    "moving_avg": 25,
    "channel_independence": 1,    # default value 0
    "down_sampling_layers": 1,
    "down_sampling_window": 2,
    "down_sampling_method": "avg",  # max, avg, conv
    "use_norm": True,
    # "decoder_input_size_multiplier":0.5,  #valid only when futr_exog_list is supported and available?
    "learning_rate": 1e-3,
    # num_lr_decays = -1,
    "early_stop_patience_steps": 10,
    "batch_size": 32,
    # step_size = 1,
    "random_seed": 7,
    "optimizer": "Adam",
    # num_workers_loader = 0,
    # drop_last_loader = False,
    # lr_scheduler=None,
    # lr_scheduler_kwargs=None,
    # NOTE: scaling will be set when instantiating `NeuralForecast`.
    # See https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/time_series_scaling.html
    # scaler_type="identity",
    "local_scaler_type": None,  # Can be ‘standard’, ‘robust’, ‘robust-iqr’, ‘minmax’ or ‘boxcox’
}

class TimeMixerModel(BaseModel):

    def __init__(self) -> None:
        super(TimeMixerModel, self).__init__()
        self.model = None
        self.nf = None
        self.val_size = None

    def restore_params(self, params: dict, **kwargs: Any) -> dict:
        return params

    def power_demand(self, args: SimpleNamespace, params: dict) -> int:
        return 2

    def gpu_threshold(self) -> Tuple[float, float]:
        return 30, 30

    def cpu_threshold(self) -> Tuple[float, float]:
        return 50, 50

    def _select_optimizer(self, **kwargs: Any) -> Tuple[Type[Optimizer], dict]:
        match kwargs["optimizer"]:
            case "Adam":
                model_optim = optim.Adam
            case "AdamW":
                model_optim = optim.AdamW
            case "SGD":
                model_optim = optim.SGD
        optim_args = {
            "lr": kwargs["learning_rate"],
            "fused": kwargs["accelerator"] in ("gpu", "auto")
            and torch.cuda.is_available(),
        }
        return model_optim, optim_args

    def _evaluate_cross_validation(self, df, metric):
        models = df.drop(columns=["unique_id", "ds", "cutoff", "y"]).columns.tolist()
        evals = []
        # Calculate loss for every unique_id and cutoff.
        for cutoff in df["cutoff"].unique():
            eval_ = evaluate(
                df[df["cutoff"] == cutoff], metrics=[metric], models=models
            )
            evals.append(eval_)
        evals = pd.concat(evals)
        evals = evals.groupby("unique_id").mean(
            numeric_only=True
        )  # Averages the error metrics for all cutoffs for every combination of model and unique_id
        evals["best_model"] = evals.idxmin(axis=1)
        return evals

    def _get_metrics(self, **kwargs: Any) -> dict:
        train_losses = self.nf.models[0].train_trajectories
        loss = min(train_losses, key=lambda x: x[1])[1]
        valid_losses = self.nf.models[0].valid_trajectories
        loss_val = (
            min(valid_losses, key=lambda x: x[1])[1]
            if len(valid_losses) > 0
            else np.nan
        )

        forecast = self.nf.predict_insample()
        forecast.reset_index(inplace=True)
        eval_mae = eval_rmse = eval_mae_val = eval_rmse_val = np.nan
        if kwargs["validate"]:
            eval_mae = self._evaluate_cross_validation(
                forecast[: -self.val_size], mae
            ).iloc[0, 0]
            eval_rmse = self._evaluate_cross_validation(
                forecast[: -self.val_size], rmse
            ).iloc[0, 0]
            eval_mae_val = self._evaluate_cross_validation(
                forecast[-self.val_size :], mae
            ).iloc[0, 0]
            eval_rmse_val = self._evaluate_cross_validation(
                forecast[-self.val_size :], rmse
            ).iloc[0, 0]
        else:
            eval_mae = self._evaluate_cross_validation(forecast, mae).iloc[0, 0]
            eval_rmse = self._evaluate_cross_validation(forecast, rmse).iloc[0, 0]

        return {
            "epoch": int(train_losses[-1][0]),
            "MAE_val": float(eval_mae_val),
            "RMSE_val": float(eval_rmse_val),
            "Loss_val": float(loss_val),
            "MAE": float(eval_mae),
            "RMSE": float(eval_rmse),
            "Loss": float(loss),
            # "device": self.device,
            # "machine": socket.gethostname(),
        }

    def trainable_on_cpu(self, **kwargs: Any) ->bool:
        return True

    def torch_cpu_ratio(self) -> float:
        return 0.35 if self.is_baseline(**self.model_args) else 0.9

    def _train(self, df: pd.DataFrame, **kwargs: Any) -> dict:
        model_config = default_params.copy()
        model_config.update(kwargs)
        # prep the parameters and df
        df.insert(0, "unique_id", "0")
        optimizer, optim_args = self._select_optimizer(**model_config)

        seed_logger = logging.getLogger("lightning_fabric.utilities.seed")
        orig_seed_log_level = seed_logger.getEffectiveLevel()
        seed_logger.setLevel(logging.FATAL)

        self.model = TimeMixer(
            h=model_config["h"],
            n_series=1,
            input_size=model_config["input_size"],
            # stat_exog_list=None,
            # hist_exog_list=None,
            # futr_exog_list=None,
            d_model=model_config["d_model"],
            d_ff=model_config["d_ff"],
            dropout=model_config["dropout"],
            e_layers=model_config["e_layers"],
            top_k=model_config["top_k"],
            decomp_method=model_config["decomp_method"],
            moving_avg=model_config["moving_avg"],
            channel_independence=model_config["channel_independence"],
            down_sampling_layers=model_config["down_sampling_layers"],
            down_sampling_window=model_config["down_sampling_window"],
            down_sampling_method=model_config["down_sampling_method"],
            use_norm=model_config["use_norm"],
            # decoder_input_size_multiplier=model_config["decoder_input_size_multiplier"],
            loss=HuberLoss(),
            # valid_loss=None,
            max_steps=model_config["max_steps"],
            learning_rate=model_config["learning_rate"],
            # num_lr_decays = -1,
            early_stop_patience_steps=(
                model_config["early_stop_patience_steps"]
                if model_config["validate"]
                else -1
            ),
            val_check_steps=int(model_config["max_steps"] / 100.0),
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
            # devices="auto",
            devices=model_config["devices"],
        )
        self.nf = NeuralForecast(
            models=[self.model],
            freq="B",
            # Scaler to apply per-serie to all features before fitting, which is inverted after predicting.
            # Can be 'standard', 'robust', 'robust-iqr', 'minmax' or 'boxcox'
            local_scaler_type=model_config["local_scaler_type"],
        )

        seed_logger.setLevel(orig_seed_log_level)

        self.val_size = min(300, int(len(df) * 0.9)) if model_config["validate"] else 0

        rank_zero_logger = logging.getLogger("lightning.pytorch.utilities.rank_zero")
        orig_log_level = rank_zero_logger.getEffectiveLevel()
        rank_zero_logger.setLevel(logging.FATAL)

        self.nf.fit(df, val_size=self.val_size)

        rank_zero_logger.setLevel(orig_log_level)

        return self._get_metrics(**model_config)

    def baseline_params(self) -> dict:
        return baseline_params

    def trim_forecast(self, forecast: pd.DataFrame) -> pd.DataFrame:
        return forecast[["ds", "yhat_n"]].copy()

    def predict(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        forecast = self.nf.predict(df)
        forecast.reset_index(drop=True, inplace=True)
        forecast.insert(forecast.columns.get_loc("ds") + 1, "y", np.nan)
        forecast.rename(columns={"TimeMixer": "yhat_n"}, inplace=True)
        return forecast

    def search_space(self, **kwargs: Any) -> str:
        return f"""dict(
            input_size=range(5, 1000+1),
            d_model=[2**w for w in range(5, 10+1)],
            d_ff=[2**w for w in range(5, 10+1)],
            dropout=uniform(0, 0.5),
            e_layers=range(4, 16+1),
            top_k=range(2, 10),
            decomp_method=["moving_avg", "dft_decomp"],
            moving_avg=range(3, 60),
            channel_independence=[0, 1],
            down_sampling_layers=range(1, 8),
            down_sampling_window=range(2, 20+1),
            down_sampling_method=["avg", "max", "conv"],
            use_norm=[True, False],
            learning_rate=loguniform(0.0001, 0.002),
            early_stop_patience_steps=range(5, 16+1),
            batch_size=[2**w for w in range(5, 8+1)],
            local_scaler_type=[None, "standard", "robust", "robust-iqr", "minmax", "boxcox"],
            topk_covar=list(range(0, {kwargs["max_covars"]}+1)),
            covar_dist=dirichlet([1.0]*{kwargs["max_covars"]}),
            optimizer=["Adam", "AdamW", "SGD"],
        )
        """

    def accept_missing_data(self) -> bool:
        return False
