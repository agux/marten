import os
import time
import uuid
import psutil
import shutil
import random
import socket
import math
import traceback

import threading
from types import SimpleNamespace, MappingProxyType

import numpy as np
import pandas as pd
import torch
from dask.distributed import get_worker, worker_client, Lock, get_client
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.varmax import VARMAX

# from statsmodels.tsa.arima.model import ARIMA
from softs.exp.exp_custom import Exp_Custom

from marten.data.worker_func import impute
from marten.utils.logger import get_logger
from marten.utils.trainer import (
    is_cuda_error,
    log_train_args,
    cuda_memory_stats,
    optimize_torch,
)
from marten.utils.holidays import get_next_trade_dates
from marten.utils.worker import (
    local_machine_power,
    TaskException,
    release_lock,
    wait_cpu,
    wait_gpu,
    restart_worker,
    workload_stage,
)

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

resource_wait_time = 5  # seconds, waiting for compute resource
lock_wait_time = 2


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


def trainable_with_cpu(model_config, n_feat):
    score = 0
    if local_machine_power() > 1:
        score += 1 if model_config["d_model"] >= 128 else 0
        score += 1 if model_config["d_core"] >= 256 else 0
        score += 1 if model_config["d_ff"] >= 256 else 0
        score += 1 if model_config["e_layers"] >= 8 else 0
        score += 0 if n_feat < 64 else 1 if n_feat < 128 else 2 if n_feat < 256 else 3
    else:
        score += 1 if model_config["d_model"] >= 64 else 0
        score += 1 if model_config["d_core"] >= 128 else 0
        score += 1 if model_config["d_ff"] >= 128 else 0
        score += 1 if model_config["e_layers"] >= 4 else 0
        score += 1 if n_feat >= 32 else 0
    return score < 3


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
        file_path = os.path.join(config["checkpoints"], setting)
        try:
            shutil.rmtree(file_path)
        except Exception as e:
            get_logger().warning("failed to remove model file: %s", file_path)

        _cleanup(model)

    return model


def train_on_gpu(
    gpu_ut,
    gpu_rt,
    model_config,
    setting,
    train,
    val,
    save_model_file,
):
    global resource_wait_time, lock_wait_time
    # large_model = is_large_model(model_config, len(train.columns))
    base_model = SOFTSPredictor.isBaseline(model_config)

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
        release_lock(lock, 2 if base_model else 7)
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


def train_on_cpu(
    cpu_util_threshold,
    mem_util_threshold,
    model_config,
    setting,
    train,
    val,
    save_model_file,
):
    global resource_wait_time, lock_wait_time
    new_config = model_config.copy()
    new_config["use_gpu"] = False
    # large_model = is_large_model(model_config, len(train.columns))
    base_model = SOFTSPredictor.isBaseline(model_config)

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

    # cpu_util_threshold = mem_util_threshold = 30 if large_model else 50
    stop_at = time.time() + resource_wait_time
    while wait_cpu(cpu_util_threshold, mem_util_threshold, stop_at):
        time.sleep(1)

    if time.time() <= stop_at:
        release_lock(lock, 2 if base_model else 7)
        # ratio = 0.9 if large_model else 0.8
        # ratio = 0.5 if base_model else 0.85
        # optimize_torch(ratio)
        m = _train(new_config, setting, train, val, save_model_file)
        return m
    else:
        release_lock(lock, 0)
        raise TimeoutError("Timeout waiting for CPU resource")


class SOFTSPredictor:

    @staticmethod
    def isBaseline(params):
        for k in baseline_config.keys():
            if k not in params or params[k] != baseline_config[k]:
                return False
        return True
        # return params == baseline_config

    @staticmethod
    def train(_df, config, model_id, random_seed, validate, save_model_file=False):
        is_base_config = SOFTSPredictor.isBaseline(config)
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
        cpu_trainable = trainable_with_cpu(model_config, n_feat)
        # ratio = 0.9 if large_model else 0.33
        # _optimize_torch(ratio)

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
        gpu_ut = (
            min(gpu_ut, 10)
            if large_model
            else gpu_ut if is_base_config else 0.3 * gpu_ut
        )
        gpu_rt = (
            min(gpu_rt, 10)
            if large_model
            else gpu_rt if is_base_config else 0.3 * gpu_rt
        )
        cpu_util_threshold = mem_util_threshold = 25 if large_model else 60

        # TODO smarter device selection: what if GPU is busy and CPU is idle?
        # swiftly detect availability between 2 devices
        # potential solution: 2 phases triage algorithm. 1.fast-track + 2. queuing for power
        gpu = model_config["use_gpu"]
        cuda_error = False
        task_key = worker.get_current_task()
        if worker.client.get_metadata([task_key, "CUDA error"], False):
            gpu = False
            cuda_error = True

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
                            # not is_base_config,
                        )
                        device = "GPU:0"
                    except Exception as e:
                        if isinstance(e, TimeoutError):
                            if not cpu_trainable:
                                continue
                            match workload_stage():
                                case "finishing":
                                    continue  # stick to GPU and avoid straggler last-task
                                case "progressing":
                                    if large_model:
                                        continue
                                    t = 0.4 * attempt**0.5
                                    if random.random() < t:
                                        continue
                                    # t = (
                                    #     0.9 * 1.0 / (attempt**0.05)
                                    #     if large_model
                                    #     else 0.8 * 1.0 / (attempt**0.2)
                                    # )
                                    # if random.random() < t:
                                    #     continue
                        elif is_cuda_error(e):
                            raise e
                        else:
                            get_logger().warning(
                                f"falling back to CPU due to error: {str(e)}"
                            )
                        # attempt = 0
                        m = train_on_cpu(
                            cpu_util_threshold,
                            mem_util_threshold,
                            model_config,
                            setting,
                            train,
                            val,
                            save_model_file,
                            # not is_base_config,
                        )
                else:
                    m = train_on_cpu(
                        cpu_util_threshold,
                        mem_util_threshold,
                        model_config,
                        setting,
                        train,
                        val,
                        save_model_file,
                        # not is_base_config,
                    )
            except TaskException as e:
                if e.task_info["restart_worker"]:
                    # TODO need to restart and re-train on CPU instead
                    restart_worker(e)
                    raise e
            except TimeoutError as e:
                # if "waiting for CPU".lower() in str(e).lower():
                # retry with GPU
                get_logger().debug("retrying: %s", str(e))
                # attempt = 0
                if not gpu and not cuda_error:
                    gpu = model_config["use_gpu"]
                # else:
                # get_logger().warning("retrying: %s", str(e))

        metrics = m.metrics
        metrics["device"] = device
        metrics["machine"] = socket.gethostname()
        return m, metrics

    @staticmethod
    def predict(model, df, region):
        global resource_wait_time, lock_wait_time

        seq_len = model.args.seq_len
        pred_len = model.args.pred_len
        input, _, scaler = _prep_df(
            df, False, seq_len, pred_len, model.args.random_seed
        )

        # worker = get_worker()
        # args = worker.args

        lock_acquired = None
        gpu_lock_key = f"""{socket.gethostname()}::GPU-{model.args.gpu}"""
        cpu_lock_key = f"""{socket.gethostname()}::CPU"""
        gpu_ut = 20
        gpu_rt = 30
        cpu_ut = 30
        cpu_rt = 50

        while True:
            lock_acquired = None
            if model.args.use_gpu:
                lock = Lock(gpu_lock_key)
                if lock.acquire(timeout=f"{lock_wait_time}s"):
                    lock_acquired = lock
                    get_logger().debug("lock acquired: %s", gpu_lock_key)
            else:
                lock = Lock(cpu_lock_key)
                if lock.acquire(timeout=f"{lock_wait_time}s"):
                    lock_acquired = lock
                    get_logger().debug("lock acquired: %s", cpu_lock_key)
            
            if lock_acquired is None:
                continue

            stop_at = time.time() + resource_wait_time
            if lock_acquired.name == gpu_lock_key:
                while wait_gpu(gpu_ut, gpu_rt, stop_at):
                    time.sleep(1)
                if time.time() <= stop_at:
                    break
            else: # CPU
                while wait_cpu(cpu_ut, cpu_rt, stop_at):
                    time.sleep(1)
                if time.time() <= stop_at:
                    if model.args.use_gpu:
                        model.args.use_gpu = False
                    # optimize_torch(0.9)
                    break
            release_lock(lock_acquired, 0)

        release_lock(lock_acquired, 5)
        try:
            forecast = model.predict(setting=model.setting, pred_data=input)
        except Exception as e:
            release_lock(lock_acquired, 0)
            if is_cuda_error(e):
                model.args.use_gpu = False
                # wait CPU resource before prediction
                lock = Lock(cpu_lock_key)
                if lock.acquire(timeout="24 hours"):
                    while wait_cpu(cpu_ut, cpu_rt):
                        time.sleep(1)
                    # optimize_torch(0.9)
                    release_lock(lock, 5)
                    forecast = model.predict(setting=model.setting, pred_data=input)
                else:
                    raise TimeoutError("timeout waiting for CPU lock")
            else:
                raise e

        yhat = None
        for fc in reversed(forecast):
            fc = scaler.inverse_transform(fc)
            if yhat is None:
                yhat = [element[-1] for element in fc]
            else:
                yhat.insert(0, fc[0, -1])

        # re-construct df with date and other columns
        last_date = input["date"].max()
        future_horizons = get_next_trade_dates(last_date, region, pred_len)
        new_dict = {
            col: [None] * len(future_horizons) for col in input.columns if col != "date"
        }
        new_dict["ds"] = future_horizons
        new_df = pd.DataFrame(new_dict)
        future_df = pd.concat([df, new_df], ignore_index=True)
        future_df = future_df.iloc[len(future_df) - len(yhat) :]
        future_df["yhat_n"] = yhat
        # convert numpy.float32 column types to float
        future_df = future_df.astype({col: float for col in future_df.select_dtypes(include=[np.float32]).columns})

        return future_df
