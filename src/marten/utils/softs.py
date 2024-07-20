import os
import time
import uuid
import psutil
import shutil
import random
import socket
import math
import traceback
import warnings
import threading
from types import SimpleNamespace, MappingProxyType

import numpy as np
import pandas as pd
import torch
from dask.distributed import get_worker, worker_client, Lock, get_client
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.varmax import VARMAX

# from statsmodels.tsa.arima.model import ARIMA
from neuralprophet import (
    set_random_seed as np_random_seed,
    set_log_level,
    NeuralProphet,
)
from softs.exp.exp_custom import Exp_Custom

from marten.utils.logger import get_logger
from marten.utils.trainer import (
    is_cuda_error,
    log_train_args,
    select_device,
    cuda_memory_stats,
)
from marten.utils.holidays import get_next_trade_dates
from marten.utils.worker import local_machine_power, TaskException

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
        "mixed_precision": True,
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
        "optimizer": "Adam",
    }
)

resource_wait_time = 30  # seconds, waiting for compute resource
lock_wait_time = 15


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def is_large_model(model_config, n_feat):
    score = 0
    if local_machine_power() > 1:
        score += 1 if model_config["d_model"] >= 256 else 0
        score += 1 if model_config["d_core"] >= 512 else 0
        score += 1 if model_config["d_ff"] >= 512 else 0
        score += 1 if model_config["e_layers"] >= 16 else 0
        score += 1 if n_feat >= 128 else 0
    else:
        score += 1 if model_config["d_model"] >= 128 else 0
        score += 1 if model_config["d_core"] >= 256 else 0
        score += 1 if model_config["d_ff"] >= 256 else 0
        score += 1 if model_config["e_layers"] >= 8 else 0
        score += 1 if n_feat >= 64 else 0
    return score >= 3

# to be deprecated
def use_gpu(model_config, n_feat, util_threshold=80, vram_threshold=80):
    use_gpu = model_config["use_gpu"]
    # larger models should wait for GPU
    should_wait = is_large_model(model_config, n_feat) if use_gpu else False
    # use_gpu = (
    #     True
    #     if should_wait
    #     else (
    #         use_gpu
    #         and torch.cuda.utilization() < util_threshold
    #         and torch.cuda.memory_usage() < vram_threshold
    #     )
    # )
    return use_gpu, should_wait


def wait_gpu(util_threshold=80, vram_threshold=80, stop_at=None):
    return (
        torch.cuda.utilization() >= util_threshold
        or torch.cuda.memory_usage() >= vram_threshold
    ) and time.time() <= stop_at


def _fit_multivariate_impute_model(input, params):
    def _fit(order, error_cov_type, cov_type):
        model = VARMAX(input, order=order, error_cov_type=error_cov_type)
        return model.fit(disp=False, cov_type=cov_type)

    ect_list = ["unstructured", "diagonal"]
    ct_list = ["opg", "robust", "robust_approx"]
    ex = None
    for ect in ect_list:
        for ct in ct_list:
            try:
                model_fit = _fit(
                    order=(params["p"], params["q"]), error_cov_type=ect, cov_type=ct
                )
                return model_fit
            except Exception as e:
                ex = e
                continue
    raise ex


def _prep_df(_df, validate, seq_len, pred_len, random_seed):
    df, _ = impute(_df, random_seed)

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
        # val_na_positions = val_data.isna()
    else:
        train_data = df
        val_data = None
        # val_na_positions = None

    # train_data_nona = train_data.dropna()
    # train_na_positions = train_data.isna()

    scaler = StandardScaler()
    # scaler.fit(train_data_nona.iloc[:, 1:])  # skip first "date" column
    # train_data_filled = train_data.ffill().bfill()
    # train_data.iloc[:, 1:] = scaler.transform(train_data_filled.iloc[:, 1:])
    train_data.iloc[:, 1:] = scaler.fit_transform(train_data.iloc[:, 1:])
    if validate:
        # val_data_filled = val_data.ffill().bfill()
        # val_data.iloc[:, 1:] = scaler.transform(val_data_filled.iloc[:, 1:])
        val_data.iloc[:, 1:] = scaler.transform(val_data.iloc[:, 1:])

    # if _df.isna().any().any():  # original dataset contains missing data
    #     # need to standardize train_data_nona first before feeding to imputation model
    #     impute_model_input = pd.DataFrame(
    #         scaler.transform(train_data_nona.iloc[:, 1:]),
    #         columns=train_data_nona.columns[1:],
    #         index=train_data_nona.index,
    #     )

    #     try:
    #         if impute_model_input.shape[1] >= 2:  # Multivariate time series
    #             model_fit = _fit_multivariate_impute_model(
    #                 impute_model_input, {"p": pred_len * 2, "q": pred_len * 2}
    #             )
    #         else:  # Univariate time series
    #             model = ARIMA(
    #                 impute_model_input, order=(5, 1, 0)
    #             )  # Adjust the order as needed
    #             model_fit = model.fit()
    #     except Exception as e:
    #         get_logger().error(
    #             "failed to fit imputation model: %s\nInput:\n%s\nTraceback:\n%s",
    #             str(e),
    #             impute_model_input,
    #             traceback.format_exc(),
    #         )
    #         raise e

    #     if train_na_positions.any().any():  # train data has missing values
    #         train_data[train_na_positions] = np.nan
    #         train_data = _impute(train_data, model_fit, True)
    #     if (
    #         val_na_positions is not None and val_na_positions.any().any()
    #     ):  # validation data has missing values
    #         val_data[val_na_positions] = np.nan
    #         val_data = _impute(val_data, model_fit, False)

    return train_data, val_data, scaler


def _statsmodels_impute(df, model_fit, in_sample):
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


def _neupro_impute(df, random_seed):
    na_col = df.columns[1]
    df.rename(columns={na_col: "y"}, inplace=True)

    np_random_seed(random_seed)
    set_log_level("ERROR")
    _optimize_torch()

    na_positions = df.isna()
    df_nona = df.dropna()
    scaler = StandardScaler()
    scaler.fit(df_nona.iloc[:, 1:])
    df_filled = df.ffill().bfill()
    df.iloc[:, 1:] = scaler.transform(df_filled.iloc[:, 1:])
    df[na_positions] = np.nan

    try:
        m = NeuralProphet(
            accelerator=select_device(True),
            # changepoints_range=1.0,
        )
        m.fit(
            df,
            progress=None,
            #   early_stopping=True,
            checkpointing=False,
        )
    except Exception as e:
        m = NeuralProphet(
            # changepoints_range=1.0,
        )
        m.fit(
            df,
            progress=None,
            #   early_stopping=True,
            checkpointing=False,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        forecast = m.predict(df)

    forecast = forecast[["ds", "yhat1"]]
    forecast["ds"] = forecast["ds"].dt.date
    forecast["yhat1"] = forecast["yhat1"].astype(float)
    forecast.rename(columns={"yhat1": na_col}, inplace=True)
    forecast.iloc[:, 1:] = scaler.inverse_transform(forecast.iloc[:, 1:])

    return forecast


def impute(df, random_seed, client=None):
    df_na = df.iloc[:, 1:].isna()

    if not df_na.any().any():
        return df, None

    na_counts = df_na.sum()
    na_cols = na_counts[na_counts > 0].index.tolist()
    na_row_indices = df[df.iloc[:, 1:].isna().any(axis=1)].index

    def _func(client):
        futures = []
        for na_col in na_cols:
            df_na = df[["ds", na_col]]
            futures.append(client.submit(_neupro_impute, df_na, random_seed))
        return client.gather(futures)

    if client is not None:
        results = _func(client)
    elif len(na_cols) > 1:
        with worker_client() as client:
            results = _func(client)
    else:
        results = [_neupro_impute(df[["ds", na_cols[0]]].copy(), random_seed)]
    imputed_df = results[0]
    for result in results[1:]:
        imputed_df = imputed_df.merge(result, on="ds", how="left")

    for na_col in na_cols:
        df[na_col].fillna(imputed_df[na_col], inplace=True)

    # Select imputed rows only
    imputed_df = imputed_df.loc[na_row_indices].copy()

    return df, imputed_df


def _optimize_torch(ratio=0.85):
    cpu_cap = (100.0 - psutil.cpu_percent(1)) / 100.0
    n_cores = float(psutil.cpu_count())
    # n_workers = max(1.0, float(num_workers()))
    # n_threads = min(int(n_cores * cpu_cap * 0.85), n_cores/n_workers)
    n_threads = max(1, int(n_cores * cpu_cap * ratio))
    torch.set_num_threads(
        n_threads
    )  # Sets the number of threads used for intraop parallelism on CPU.
    get_logger().debug(
        "machine: %s, cpu_cap: %s, n_cores: %s optimizing torch CPU thread: %s",
        socket.gethostname(),
        round(cpu_cap, 3),
        int(n_cores),
        n_threads,
    )

    # Enable cuDNN auto-tuner
    # torch.backends.cudnn.benchmark = True


def _train(config, setting, train, val, save_model_file):
    torch.cuda.empty_cache()
    model = Exp_Custom(SimpleNamespace(**config))

    def _cleanup(model):
        if model is not None:
            model.cleanup()
            del model
            torch.cuda.empty_cache()

    try:
        model.train(
            setting=setting,
            train_data=train,
            vali_data=val,
        )
    except Exception as e:
        _cleanup(model)
        raise e
    if val is not None:
        model.test(setting=setting, test_data=val)  # collect validation metrics
    if not save_model_file:
        shutil.rmtree(os.path.join(config["checkpoints"], setting))

    _cleanup(model)

    return model


def restart_worker(exception):
    worker = get_worker()
    if worker is not None:
        get_logger().warning(
            "trying to restart worker %s due to CUDA error: %s.\n%s",
            worker.address,
            str(exception),
            cuda_memory_stats(),
        )
        get_client().restart_workers(
            workers=[worker.address], timeout=600, raise_for_error=False
        )


def train_on_gpu(
    gpu_ut, gpu_rt, model_config, setting, train, val, save_model_file
):
    global resource_wait_time, lock_wait_time
    # large_model = is_large_model(model_config, len(train.columns))

    lock_key = f"""{socket.gethostname()}::GPU-{model_config["gpu"]}"""

    lock = Lock(lock_key)
    lock_wait_start = time.time()
    lock_acquired = False
    while time.time() - lock_wait_start <= resource_wait_time:
        if lock.acquire(timeout=f"{lock_wait_time}s"):
            lock_acquired = True
            get_logger().debug("lock acquired: %s", lock_key)
            break
    if not lock_acquired:
        # get_logger().debug("Timeout waiting for GPU lock: %s", lock_key)
        raise TimeoutError(f"Timeout waiting for GPU lock: {lock_key}")

    stop_at = time.time() + resource_wait_time
    while wait_gpu(gpu_ut, gpu_rt, stop_at):
        time.sleep(1)
    if time.time() <= stop_at:
        release_lock(lock)
        try:
            m = _train(model_config, setting, train, val, save_model_file)
            return m
        except Exception as e:
            release_lock(lock, 0)
            if is_cuda_error(e):
                raise TaskException(
                    f"CUDA error: {str(e)} Memory stats: {cuda_memory_stats()}",
                    worker={"address": get_worker().address},
                    restart_worker=True,
                )
            raise e
    else:
        release_lock(lock, 0)
        raise TimeoutError("Timeout waiting for GPU resource")


def train_on_cpu(model_config, setting, train, val, save_model_file):
    global resource_wait_time, lock_wait_time
    new_config = model_config.copy()
    new_config["use_gpu"] = False
    large_model = is_large_model(model_config, len(train.columns))

    lock_key = f"{socket.gethostname()}::CPU"
    lock = Lock(lock_key)
    lock_wait_start = time.time()
    lock_acquired = False
    while time.time() - lock_wait_start <= resource_wait_time:
        if lock.acquire(timeout=f"{lock_wait_time}s"):
            lock_acquired = True
            get_logger().debug("lock acquired: %s", lock_key)
            break
    if not lock_acquired:
        raise TimeoutError(f"Timeout waiting for CPU lock {lock_key}")

    cpu_util_threshold = mem_util_threshold = 30 if large_model else 50
    stop_at = time.time() + resource_wait_time
    cpu_util = psutil.cpu_percent(1)
    mem_util = psutil.virtual_memory().percent
    while (
        cpu_util >= cpu_util_threshold or mem_util >= mem_util_threshold
    ) and time.time() <= stop_at:
        time.sleep(1)
        cpu_util = psutil.cpu_percent(1)
        mem_util = psutil.virtual_memory().percent

    if time.time() <= stop_at:
        release_lock(lock)
        ratio = 0.9 if large_model else 0.8
        _optimize_torch(ratio)
        m = _train(new_config, setting, train, val, save_model_file)
        return m
    else:
        release_lock(lock, 0)
        raise TimeoutError("Timeout waiting for CPU resource")


def release_lock(lock, after=10):
    if lock is None:
        return

    def _release():
        get_logger().debug("lock %s will be released in %s seconds", lock.name, after)
        time.sleep(after)
        try:
            lock.release()
            get_logger().debug("lock %s released", lock.name)
        except Exception as e:
            get_logger().warning("exception releasing lock %s: %s", lock.name, str(e))

    threading.Timer(after, _release).start()


class SOFTSPredictor:

    @staticmethod
    def isBaseline(params):
        return params == baseline_config

    @staticmethod
    def train(_df, config, model_id, random_seed, validate, save_model_file=False):
        df = _df.copy()
        worker = get_worker()
        args = worker.args

        set_random_seed(random_seed)
        model_config = default_config.copy()
        model_config.update(config)
        model_config["random_seed"] = random_seed

        if "d_model_d_ff" in model_config:
            model_config["d_model"] = model_config["d_model_d_ff"]
            model_config["d_ff"] = model_config["d_model_d_ff"]
            model_config.pop("d_model_d_ff")

        n_feat = len(df.columns)
        large_model = is_large_model(model_config, n_feat)
        ratio = 0.9 if large_model else 0.33
        _optimize_torch(ratio)

        train, val, _ = _prep_df(
            df, validate, model_config["seq_len"], model_config["pred_len"], random_seed
        )

        if getattr(args, "log_train_args", False):
            log_train_args(
                _df, train, val, model_config, random_seed, validate, save_model_file
            )

        setting = os.path.join(model_id, str(uuid.uuid4()))

        device = "CPU"
        gpu_ut = getattr(args, "gpu_util_threshold", 80)
        gpu_rt = getattr(args, "gpu_ram_threshold", 80)
        gpu_ut = gpu_ut * 0.5 if large_model else gpu_ut
        gpu_rt = gpu_rt * 0.5 if large_model else gpu_rt

        # TODO smarter device selection: what if GPU is busy and CPU is idle?
        # swiftly detect availability between 2 devices
        # potential solution: 2 phases triage algorithm. 1.fast-track + 2. queuing for power
        gpu = model_config["use_gpu"]
        get_logger().info("futures info from worker: %s", worker.client.futures)
        # worker.data
        # worker.client
        # worker.client.futures
        # worker.client.get_metadata
        # worker.client.processing
        # worker.client.set_metadata

        m = None
        attempt = 0
        while m is None:
            attempt += 1
            try:
                if gpu:
                    try:
                        m = train_on_gpu(
                            gpu_ut,
                            gpu_rt,
                            model_config,
                            setting,
                            train,
                            val,
                            save_model_file,
                        )
                        device = "GPU:0"
                    except Exception as e:
                        if is_cuda_error(e):
                            raise e
                        elif isinstance(e, TimeoutError):
                            t = (
                                0.9 * 1.0 / (attempt**0.05)
                                if is_large_model
                                else 0.8 * 1.0 / (attempt**0.2)
                            )
                            if random.random() < t:
                                continue
                        get_logger().warning(
                            f"falling back to CPU due to error: {str(e)}"
                        )
                        m = train_on_cpu(
                            model_config, setting, train, val, save_model_file
                        )
                else:
                    m = train_on_cpu(model_config, setting, train, val, save_model_file)
            except TimeoutError as e:
                # if "waiting for CPU".lower() in str(e).lower():
                # retry with GPU
                get_logger().debug("retrying: %s", str(e))
                if not gpu:
                    gpu = model_config["use_gpu"]
                # else:
                # get_logger().warning("retrying: %s", str(e))

        metrics = m.metrics
        metrics["device"] = device
        metrics["machine"] = socket.gethostname()
        return m, metrics

    @staticmethod
    def predict(model, df, region):
        seq_len = model.args.seq_len
        pred_len = model.args.pred_len
        input, _, scaler = _prep_df(
            df, False, seq_len, pred_len, model.args.random_seed
        )
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
