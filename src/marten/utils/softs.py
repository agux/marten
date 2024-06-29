import os
import uuid
import psutil
import shutil
import random
import traceback
from types import SimpleNamespace, MappingProxyType

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.varmax import VARMAX
from tenacity import (
    stop_after_attempt,
    wait_exponential,
    Retrying,
    retry_if_exception,
    RetryError,
)
from softs.exp.exp_custom import Exp_Custom

from marten.utils.logger import get_logger
from marten.utils.trainer import should_retry, log_retry
from marten.utils.holidays import get_next_trade_dates

# Immutable / constant
default_config = MappingProxyType({
    "features": "MS",
    "freq": "B",
    "model": "SOFTS",
    "checkpoints": "./softs_checkpoints/",
    "loss_func": "huber",
    "gpu": "0",  # TODO support smart GPU selection if multiple GPU is available
    "save_model": True,
    "num_workers": 0,
    "predict_all": True,
})

# Immutable / constant
baseline_config = MappingProxyType({
    "seq_len": 5,
    "d_model": 32,
    "d_core": 16,
    "d_ff": 32,
    "e_layers": 2,
    "learning_rate": 3e-4,  # 0.0003
    "lradj": "cosine",
    "patience": 3,
    "batch_size": 16,
    "dropout": 0.0,
    "activation": "gelu",
    "use_norm": False,
})


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


class SOFTSPredictor:

    @staticmethod
    def isBaseline(params):
        return params == baseline_config

    @staticmethod
    def _impute(df, model_fit):
        data_filled = df.copy()
        forecast = model_fit.get_forecast(steps=len(df))
        for col in df.columns[1:]:
            if df[col].isna().any():
                data_filled[col].fillna(forecast.predicted_mean[col], inplace=True)
        return data_filled

    @staticmethod
    def _prep_df(_df, validate, seq_len):
        df = _df.copy()

        if df.columns[0] == "ds":
            df.rename(columns={"ds": "date"}, inplace=True)

        if "y" in df.columns:
            y_position = df.columns.get_loc("y")
            if y_position != len(df.columns) - 1:
                # Move "y" column to the last position
                cols = list(df.columns)
                cols.append(cols.pop(y_position))
                df = df[cols]

        if validate:
            n = len(df)
            end = int(n * 0.9)
            train_data = df.iloc[:end]
            val_data = df.iloc[end - seq_len :]
            val_na_positions = val_data.isna()
        else:
            train_data = df
            val_data = None
            val_na_positions = []

        train_data_nona = train_data.dropna()
        train_na_positions = train_data.isna()

        scaler = StandardScaler()
        scaler.fit(train_data_nona.iloc[:, 1:])  # skip first "date" column
        train_data_filled = train_data.ffill().bfill()
        train_data.iloc[:, 1:] = scaler.transform(train_data_filled.iloc[:, 1:])
        if validate:
            val_data_filled = val_data.ffill().bfill()
            val_data.iloc[:, 1:] = scaler.transform(val_data_filled.iloc[:, 1:])

        if len(df.isna()) > 0:  # dataset contains missing data
            model = VARMAX(train_data_nona[:, 1:], order=(1, 1))
            model_fit = model.fit(disp=False)

            if len(train_na_positions) > 0:
                train_data[train_na_positions] = np.nan
                train_data = SOFTSPredictor._impute(train_data, model_fit)
            if len(val_na_positions) > 0:
                val_data[val_na_positions] = np.nan
                val_data = SOFTSPredictor._impute(val_data, model_fit)

        return train_data, val_data, scaler

    @staticmethod
    def train(df, config, model_id, random_seed, validate, save_model_file=False):
        set_random_seed(random_seed)
        model_config = default_config.copy().update(config)

        num_cores = psutil.cpu_count(logical=False)
        torch.set_num_threads(num_cores)

        train, val, _ = SOFTSPredictor._prep_df(df, validate, config.seq_len)
        setting = os.path.join(model_id, str(uuid.uuid4()))

        def _train(config):
            Exp = Exp_Custom(SimpleNamespace(**config))
            Exp.train(
                setting=setting,
                train_data=train,
                vali_data=val,
            )
            if val is not None:
                Exp.test(setting=setting, test_data=val)  # collect validation metrics
            if not save_model_file:
                shutil.rmtree(os.path.join(config["checkpoints"], setting))
            return Exp

        try:
            for attempt in Retrying(
                stop=stop_after_attempt(2),
                wait=wait_exponential(multiplier=1, max=10),
                retry=retry_if_exception(should_retry),
                before_sleep=log_retry,
            ):
                with attempt:
                    m = _train(model_config)
                    return m, m.metrics
        except RetryError as e:
            new_config = model_config.copy()
            new_config["use_gpu"] = False
            full_traceback = traceback.format_exc()

            if "OutOfMemoryError" in str(e) or "out of memory" in full_traceback:
                # final attempt to train on CPU
                get_logger().warning("falling back to CPU due to OutOfMemoryError")
            else:
                get_logger().warning(f"falling back to CPU due to RetryError: {str(e)}")

            m = _train(new_config)
            return m, m.metrics

    @staticmethod
    def predict(model, df, region):
        seq_len = model.args.seq_len
        pred_len = model.args.pred_len
        input, _, scaler = SOFTSPredictor._prep_df(df, False, seq_len)
        forecast = model.predict(setting=model.setting, pred_data=input)
        yhat = None
        for fc in reversed(forecast):
            fc = scaler.inverse_transform(fc)
            if yhat is None:
                yhat = [element[-1] for element in fc]
            else:
                yhat.insert(0, fc[0, -1])

        # re-construct df with date and other columns
        last_date = df["ds"].max()
        future_horizons = get_next_trade_dates(last_date, region, pred_len)
        new_dict = {col: [None] * len(future_horizons) for col in df.columns if col != "ds"}
        new_dict["ds"] = future_horizons
        new_df = pd.DataFrame(new_dict)
        future_df = pd.concat([df, new_df], ignore_index=True)
        future_df = future_df[len(future_df)-len(yhat):]
        future_df["yhat_n"] = yhat
        
        return future_df
