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
from dask.distributed import get_worker, get_client
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.arima.model import ARIMA
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
from marten.utils.worker import num_workers

# Immutable / constant
default_config = MappingProxyType(
    {
        "features": "MS",
        "freq": "B",
        "model": "SOFTS",
        "checkpoints": "./softs_checkpoints/",
        "loss_func": "huber",
        "gpu": "0",  # TODO support smart GPU selection if multiple GPU is available
        "save_model": True,
        "num_workers": 0,
        "predict_all": True,
    }
)

# Immutable / constant
baseline_config = MappingProxyType(
    {
        "seq_len": 5,
        "d_model": 32,
        "d_core": 16,
        "d_ff": 32,
        "e_layers": 2,
        "learning_rate": 1e-4,  # 0.0001
        "lradj": "cosine",
        "patience": 3,
        "batch_size": 16,
        "dropout": 0.0,
        "activation": "gelu",
        "use_norm": False,
    }
)


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def use_gpu(use_gpu, util_threshold=80, vram_threshold=80):
    return (
        True
        if use_gpu
        and torch.cuda.utilization() < util_threshold
        and torch.cuda.memory_usage() < vram_threshold
        else False
    )

def _fit_multivariate_impute_model(input):
    def _fit(order, error_cov_type, cov_type):
        model = VARMAX(input, order=order, error_cov_type=error_cov_type)
        return model.fit(disp=False, cov_type=cov_type)
    ect_list = ["unstructured", "diagonal"]
    ct_list = ["robust", "robust_approx"]
    ex = None
    for ect in ect_list:
        for ct in ct_list:
            try:
                model_fit = _fit(order=(1,1), error_cov_type=ect, cov_type=ct)
                return model_fit
            except Exception as e:
                ex = e
                continue
    raise ex

def _prep_df(_df, validate, seq_len):
    df = _df.copy()

    if "ds" in df.columns:
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
        val_na_positions = None

    train_data_nona = train_data.dropna()
    train_na_positions = train_data.isna()

    scaler = StandardScaler()
    scaler.fit(train_data_nona.iloc[:, 1:])  # skip first "date" column
    train_data_filled = train_data.ffill().bfill()
    train_data.iloc[:, 1:] = scaler.transform(train_data_filled.iloc[:, 1:])
    if validate:
        val_data_filled = val_data.ffill().bfill()
        val_data.iloc[:, 1:] = scaler.transform(val_data_filled.iloc[:, 1:])

    if _df.isna().any().any():  # original dataset contains missing data
        # need to standardize train_data_nona first before feeding to imputation model
        impute_model_input = pd.DataFrame(
            scaler.transform(train_data_nona.iloc[:, 1:]),
            columns=train_data_nona.columns[1:],
            index=train_data_nona.index,
        )

        try:
            if impute_model_input.shape[1] >= 2:  # Multivariate time series
                model_fit = _fit_multivariate_impute_model(impute_model_input)
            else:  # Univariate time series
                model = ARIMA(
                    impute_model_input, order=(5, 1, 0)
                )  # Adjust the order as needed
                model_fit = model.fit()
        except Exception as e:
            get_logger().error(
                "failed to fit imputation model: %s\nInput:\n%s\nTraceback:\n%s",
                str(e),
                impute_model_input,
                traceback.format_exc(),
            )
            raise e

        if train_na_positions.any().any():  # train data has missing values
            train_data[train_na_positions] = np.nan
            train_data = _impute(train_data, model_fit, True)
        if (
            val_na_positions is not None and val_na_positions.any().any()
        ):  # validation data has missing values
            val_data[val_na_positions] = np.nan
            val_data = _impute(val_data, model_fit, False)

    return train_data, val_data, scaler


def _impute(df, model_fit, in_sample):
    data_filled = df.copy()
    # if df.shape[1] > 2:  # Multivariate time series
    if in_sample:
        forecast = model_fit.get_prediction(
            start=data_filled.index[0],
            end=data_filled.index[-1],
        )
    else:
        forecast = model_fit.get_forecast(steps=len(df))
    predicted_mean = forecast.predicted_mean
    predicted_mean.index = data_filled.index
    for col in df.columns[1:]:
        if df[col].isna().any():
            data_filled[col].fillna(predicted_mean[col], inplace=True)
    # else:  # Univariate time series
    #     for col in df.columns[1:]:
    #         if df[col].isna().any():
    #             forecast = model_fit.predict(
    #                 start=len(df[col].dropna()), end=len(df) - 1, dynamic=False
    #             )
    #             data_filled[col].fillna(
    #                 pd.Series(forecast, index=df.index[df[col].isna()]),
    #                 inplace=True,
    #             )
    return data_filled


class SOFTSPredictor:

    @staticmethod
    def isBaseline(params):
        return params == baseline_config

    @staticmethod
    def train(df, config, model_id, random_seed, validate, save_model_file=False):
        worker = get_worker()
        args = worker.args

        set_random_seed(random_seed)
        model_config = default_config.copy()
        model_config.update(config)

        # n_cores = float(psutil.cpu_count())
        n_cores = psutil.cpu_count(logical=False)
        n_workers = float(num_workers())
        torch.set_num_threads(max(1, int(n_cores / n_workers)))

        train, val, _ = _prep_df(df, validate, model_config["seq_len"])
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
            if not use_gpu(
                model_config["use_gpu"],
                getattr(args, "gpu_util_threshold", None),
                getattr(args, "gpu_ram_threshold", None),
            ):
                new_config = model_config.copy()
                new_config["use_gpu"] = False
                m = _train(new_config)
                return m, m.metrics

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
        input, _, scaler = _prep_df(df, False, seq_len)
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
        new_dict = {
            col: [None] * len(future_horizons) for col in df.columns if col != "ds"
        }
        new_dict["ds"] = future_horizons
        new_df = pd.DataFrame(new_dict)
        future_df = pd.concat([df, new_df], ignore_index=True)
        future_df = future_df.iloc[len(future_df) - len(yhat) :]
        future_df["yhat_n"] = yhat

        return future_df
