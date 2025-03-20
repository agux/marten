import warnings

# Ignore the specific FutureWarning about Series.view deprecation
warnings.filterwarnings(
    action="ignore",
    message=".*Series.view is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    action="ignore",
    message="The epoch parameter in `scheduler.step\\(\\)` was not necessary.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Trying to infer the `batch_size` from an ambiguous collection.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"You called `self\.log\(.*but have no logger configured.*",
    category=UserWarning,
)

from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import Any, Tuple, List, Type
from dotenv import load_dotenv
import os
import time
import socket
import threading
import numpy as np
import pandas as pd
import logging
import warnings
import shutil

from cpuinfo import get_cpu_info

from neuralprophet import (
    set_random_seed as np_random_seed,
    set_log_level,
    NeuralProphet,
)

import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.profiler import profile, ProfilerActivity

from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utilsforecast.losses import _pl_agg_expr, _base_docstring, mae, rmse
from utilsforecast.evaluation import evaluate
from utilsforecast.compat import DataFrame, pl

from dask.distributed import get_worker, Lock

from marten.utils.logger import get_logger
from marten.utils.trainer import is_cuda_error, huber_loss
from marten.utils.system import bool_envar, FileLock
from marten.utils.worker import (
    release_lock,
    wait_gpu,
    wait_mps,
    restart_worker,
    # workload_stage,
    gpu_util,
    mps_util,
)


# @_base_docstring
# def huber_loss(
#     df: DataFrame,
#     models: List[str],
#     delta: float = 1.0,
#     id_col: str = "unique_id",
#     target_col: str = "y",
# ) -> DataFrame:
#     r"""
#     Huber Loss

#     Huber Loss is a loss function used in robust regression that is less sensitive to outliers in data than the squared error loss.
#     It is defined as:
#     L_{\delta}(a) =
#         0.5 * a^2                  if |a| <= delta
#         delta * (|a| - 0.5 * delta) otherwise
#     where a = y - y_hat and delta is a threshold parameter.
#     """
#     if isinstance(df, pd.DataFrame):

#         def huber(a):
#             return np.where(
#                 np.abs(a) <= delta, 0.5 * a**2, delta * (np.abs(a) - 0.5 * delta)
#             )

#         res = (
#             df[models]
#             .sub(df[target_col], axis=0)
#             .apply(huber)
#             .groupby(df[id_col], observed=True)
#             .mean()
#         )
#         res.index.name = id_col
#         res = res.reset_index()
#     else:

#         def gen_expr(model):
#             a = pl.col(target_col).sub(pl.col(model))
#             return (
#                 pl.when(a.abs() <= delta)
#                 .then(0.5 * a.pow(2))
#                 .otherwise(delta * (a.abs() - 0.5 * delta))
#                 .alias(model)
#             )

#         res = _pl_agg_expr(df, models, id_col, gen_expr)
#     return res


class _dummyLock:
    def __init__(self, max_leases, name):
        self.max_leases = max_leases
        self.name = name
        self.locked = False
        # self.lock = None

    def acquire(self, timeout):
        return True

    def release(self):
        pass

    def get_value(self):
        return 0


class BaseModel(ABC):

    def __init__(self) -> None:
        super().__init__()
        self.device = None
        self.model_args = None
        self.accelerator_lock = None
        self.locks = {
            "gpu": FileLock(name=f"""{socket.gethostname()}_marten_GPU-auto.lock"""),
            "profiler": FileLock(name=f"""{socket.gethostname()}_marten_profiler.lock""")
        }
        self.profiler = None
        self.trainLogger = None

        load_dotenv()

        self.device_lock_release_delay = float(
            os.getenv("DEVICE_LOCK_RELEASE_DELAY", 2)
        )
        self.device_lock_release_delay_large = float(
            os.getenv("DEVICE_LOCK_RELEASE_DELAY_LARGE", 5)
        )
        self.resource_wait_time = float(
            os.getenv("RESOURCE_WAIT_TIME", 5)
        )  # seconds, waiting for compute resource.
        self.lock_wait_time = float(os.getenv("LOCK_WAIT_TIME", "2"))

        self.profile_folder = "profiles"
        self.profile_enabled = bool_envar("model_profile_enabled", False)
        self.profile_memory = bool_envar("model_profile_memory", False)
        self.zentorch_enabled = bool_envar("zentorch_enabled", False)
        if self.zentorch_enabled:
            self._enable_zentorch()

        logging.getLogger("lightning").setLevel(logging.WARNING)
        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
        logging.getLogger("lightning_utilities").setLevel(logging.WARNING)

    def _enable_zentorch(self):
        info = get_cpu_info()
        cpu_brand = info.get("brand_raw", "").upper()
        self.zentorch_enabled = "AMD" in cpu_brand

    def _init_profiler(self):
        self.profiler = None
        if self.model_args["accelerator"] != "cpu" or not self.profile_enabled:
            return

        # if "profiler" not in self.locks:
        # self.locks["profiler"] = Lock(name=f"""{socket.gethostname()}__profiler""")

        if not self.locks["profiler"].acquire(timeout=2):
            return

        activities = [
            ProfilerActivity.CPU,
        ]
        # if torch.cuda.is_available():
        #     activities.append(ProfilerActivity.CUDA)

        self.profiler = profile(
            activities=activities,
            # record_shapes=True,
            profile_memory=self.profile_memory,
            with_stack=True,  # Enables stack trace recording
            with_flops=True,
            with_modules=True,
        )
        os.makedirs(self.profile_folder, exist_ok=True)

    def is_baseline(self, **kwargs: Any) -> bool:
        """
        Check if the given parameters are model-specific baseline parameters.

        Args:
            kwargs (Any): model parameters.

        Returns:
            bool: True if the parameters comprise a baseline model, False otherwise.
        """
        baseline_config = self.baseline_params()
        for k in baseline_config.keys():
            if k not in kwargs or kwargs[k] != baseline_config[k]:
                return False
        return True

    def release_accelerator_lock(self, delay=0) -> None:
        release_lock(self.accelerator_lock, delay)
        self.accelerator_lock = None

    def _select_optimizer(self, **kwargs: Any) -> Tuple[Type[Optimizer], dict]:
        match kwargs["optimizer"]:
            case "Adam":
                model_optim = optim.Adam
            case "AdamW":
                model_optim = optim.AdamW
            case "SGD":
                model_optim = optim.SGD
        optim_args = {
            # "lr": kwargs["learning_rate"],
            "fused": kwargs["accelerator"] in ("gpu", "auto")
            # and torch.cuda.is_available(),
        }
        return model_optim, optim_args

    def _check_cpu(self) -> bool:
        return self.trainable_on_cpu(**self.model_args)
        # if not self.trainable_on_cpu(**self.model_args):
        #     return False

        # match workload_stage():
        #     case "finishing":
        #         return False  # stick to GPU and avoid straggler last-task
        #     # case "starting":
        #     #     return True
        #     case _:
        #         return True

    def _lock_accelerator(self, accelerator) -> str:
        if accelerator == "cpu":
            return accelerator
        # get_logger().info("locking accelerator: %s", self.locks)

        def lock_device(device):
            nonlocal accelerator, self
            if self.accelerator_lock is None and accelerator in (device, "auto"):
                lock = self.locks[device]
                w = get_worker()
                get_logger().debug(
                    "[worker#%s] acquiring lock: %s", w.name, lock.name
                )
                acquired = lock.acquire(timeout=self.lock_wait_time)
                if acquired:
                    self.accelerator_lock = lock
                    get_logger().debug("[worker#%s] lock acquired: %s", w.name, self.accelerator_lock.name)

        gpu_ut, gpu_rt = self.gpu_threshold()
        while True:
            self.accelerator_lock = None

            if accelerator == "mps":
                dam, cam = mps_util()
                lock_device("mps")
            else:
                gu, _ = gpu_util()
                if gu > 0 and self._check_cpu():
                    return "cpu"
                else:
                    if self.locks and "gpu" in self.locks.keys():
                        get_logger().debug("GPU Util:%s, trying GPU lock first", gu)
                        lock_device("gpu")
                    else:
                        return "gpu"

            if not self.accelerator_lock:
                if self._check_cpu():
                    return "cpu"
                else:
                    continue

            stop_at = time.time() + self.resource_wait_time
            if accelerator == "mps":
                while wait_mps(stop_at):
                    threading.Event().wait(0.5)
                    # time.sleep(0.5)
            else:
                while wait_gpu(gpu_ut, gpu_rt, stop_at):
                    threading.Event().wait(0.2)
                    # time.sleep(0.2)

            now = time.time()
            if now <= stop_at:
                return accelerator

            # wait GPU idleness timeout. see if we can try CPU
            if accelerator != "mps" and self._check_cpu():
                return "cpu"
            else:
                self.release_accelerator_lock()

    def _select_accelerator(self) -> str:
        accelerator = self.model_args["accelerator"].lower()
        gpu_proc = self.model_args.get("gpu_proc", 2)
        # if accelerator == "cpu":
        #     return accelerator
        is_baseline = self.is_baseline(**self.model_args)
        if accelerator in ("mps", "auto") and torch.backends.mps.is_available():
            accelerator = "mps"
        # elif accelerator in ("gpu", "auto") and not self.model_args["cuda_available"]:
        #     accelerator = "cpu"
        elif accelerator in ("gpu", "auto"):
            accelerator = "gpu"
            # if not is_baseline:
            # self.release_accelerator_lock()
            # TODO: confirm if we need to re-create the Lock instance each time to avoid getting stuck
            # self.locks["gpu"] = Lock(name=f"""{socket.gethostname()}__GPU-auto""")

        # gpu or auto
        if accelerator == "gpu":
            worker = get_worker()
            if worker is not None:
                # FIXME: raises the following error after hours of running:
                # distributed.comm.core.CommClosedError: in <TCP (closed) ConnectionPool.get_metadata
                # local=tcp://<worker_ip>:<worker_port> remote=tcp://<scheduler_ip>:<port>>:
                # TimeoutError: [Errno 110] Connection timed out

                # task_key = worker.get_current_task()
                # if worker.client.get_metadata([task_key, "CUDA error"], False):
                #     accelerator = "cpu"
                if is_baseline and int(worker.name) > gpu_proc:
                    accelerator = "cpu"

        if accelerator != "cpu":
            self.release_accelerator_lock()
            accelerator = self._lock_accelerator(accelerator)
            self.release_accelerator_lock(
                self.device_lock_release_delay
                if is_baseline
                else self.device_lock_release_delay_large
            )
        return accelerator

    def _evaluate_cross_validation(self, df, metric):
        models = df.drop(
            columns=["index", "unique_id", "ds", "cutoff", "y"]
        ).columns.tolist()
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
        model_name = str(self.nf.models[0])
        train_trajectories = self.nf.models[0].train_trajectories
        if kwargs["accelerator"] == "gpu" and not self.is_baseline(**self.model_args):
            # without exclusive lock, it may fail due to parallel overuse and insufficient GPU memory.
            self.release_accelerator_lock()
            while True:
                if self._lock_accelerator("gpu") == "gpu":
                    break
                else:
                    threading.Event().wait(0.5)
        try:
            forecast = self.nf.predict_insample(step_size=kwargs["h"])
        except Exception as e:
            get_logger().error("failed to predict insample: %s", e, exc_info=True)
            raise e
        finally:
            self.release_accelerator_lock()
            # torch.cuda.empty_cache()

        forecast = forecast.dropna(subset=[model_name])
        forecast.reset_index(inplace=True)

        get_logger().debug(
            "forecast for %s:\n%s", get_worker().get_current_task(), forecast.tail()
        )

        loss = loss_val = eval_mae = eval_rmse = eval_mae_val = eval_rmse_val = np.nan
        if kwargs["validate"]:
            train_set = forecast[: -self.val_size]
            val_set = forecast[-self.val_size :]

            loss = huber_loss(train_set["y"], train_set[model_name])
            eval_mae = mean_absolute_error(
                train_set["y"], train_set[model_name]
            )
            eval_rmse = np.sqrt(
                mean_squared_error(train_set["y"], train_set[model_name])
            )

            loss_val = huber_loss(val_set["y"], val_set[model_name])
            eval_mae_val = mean_absolute_error(
                val_set["y"], val_set[model_name]
            )
            eval_rmse_val = np.sqrt(
                mean_squared_error(val_set["y"], val_set[model_name])
            )
        else:
            loss = huber_loss(forecast["y"], forecast[model_name])
            eval_mae = mean_absolute_error(
                forecast["y"], forecast[model_name]
            )
            eval_rmse = np.sqrt(
                mean_squared_error(forecast["y"], forecast[model_name])
            )

        # shutil.rmtree(self.csvLogger.log_dir)

        return {
            "epoch": int(train_trajectories[-1][0]),
            "MAE_val": float(eval_mae_val),
            "RMSE_val": float(eval_rmse_val),
            "Loss_val": float(loss_val),
            "MAE": float(eval_mae),
            "RMSE": float(eval_rmse),
            "Loss": float(loss),
            # "device": self.device,
            # "machine": socket.gethostname(),
        }

    @abstractmethod
    def restore_params(self, params: dict, **kwargs: Any) -> dict:
        """
        Restore and complete model params from foundation params read from persistent storage.

        Args:
            params (dict): Foundation params read from persistent storage.
            **kwargs (Any): Additional context parameters.

        Returns:
            dict: the restored model params.
        """
        pass

    @abstractmethod
    def power_demand(self, args: SimpleNamespace, params: dict) -> int:
        """
        Estimates numeric indicator for the compute power demand of the model based on given program args and model params.

        Args:
            args (SimpleNamespace): program arguments.
            params (dict): model params.

        Returns:
            int: typically, 2 for larger models and 1 for regular models.
        """
        pass

    def configure_torch(self):
        # if not is_baseline:
        #     n_workers = num_workers()
        #     cpu_count = psutil.cpu_count(logical=True)
        #     num_threads = math.ceil(cpu_count / n_workers)
        #     torch.set_num_threads(num_threads)

        # n_workers = num_workers() * 0.9
        # cpu_count = psutil.cpu_count(logical=not is_baseline)
        # quotient = cpu_count / n_workers
        # floor = int(cpu_count // n_workers)
        # mod = cpu_count % n_workers
        # num_threads = math.ceil(quotient) if random.random() < mod / n_workers else floor
        num_threads = self.torch_num_threads()

        # torch.set_num_interop_threads(num_threads)
        # NOTE cannot set num_interop_threads here otherwise
        # RuntimeError: Error: cannot set number of interop threads after parallel work has started or set_num_interop_threads called

        torch.set_num_threads(num_threads)
        return torch.get_num_threads()
        # return optimize_torch_on_cpu(self.torch_cpu_ratio())

    def _init_train_logger(self):
        if self.trainLogger:
            shutil.rmtree(self.trainLogger.log_dir)
        else:
            self.trainLogger = TensorBoardLogger(
                save_dir="lightning_logs",
                name="",
                version=get_worker().name,
            )

    def train(self, df: pd.DataFrame, **kwargs: Any) -> dict:
        """
        Select the proper accelerator and train the model with the given data.

        Args:
            df (pandas.Dataframe): Time series dataframe of the dependent variable, optionally paired with covariate(s).
            **kwargs (Any): Additional context parameters.

        Returns:
            metrics (dict): The metrics of the training process. It must contain these fields
                - epoch: 0-based;
                - MAE_val: Mean Average Error for validation set;
                - RMSE_val: Root Mean Squared Error for validation set;
                - Loss_val: indicative loss value for validation set;
                - MAE: Mean Average Error for training set;
                - RMSE: Root Mean Squared Error for training set;
                - Loss: indicative loss value for training set;
                - device: e.g. GPU:0, CPU, etc.
                - machine: which machine the model is trained on. Can be obtained from socket.gethostname()
        """
        # get_logger().info("training : %s", kwargs["locks"])
        # self.locks = kwargs["locks"]
        df = df.copy()
        df.insert(0, "unique_id", "0")
        self.model_args = kwargs
        accelerator = self._select_accelerator()
        kwargs["accelerator"] = accelerator
        cpu_cores = None
        if accelerator == "gpu":
            kwargs["devices"] = "auto"
        else:
            if accelerator == "cpu":
                cpu_cores = self.configure_torch()
            # NOTE: `devices` selected with `CPUAccelerator` should be an int > 0.
            kwargs["devices"] = 1
        self.model_args = kwargs
        start_time = time.time()
        # loop = asyncio.get_running_loop()
        # loop = get_worker().io_loop
        self._init_profiler()
        self._init_train_logger()
        try:
            get_logger().debug("training with kwargs: %s", kwargs)
            # model_config = asyncio.wait_for(
            #     loop.run_in_executor(None, self._train, df, **kwargs), None
            # )
            if self.profiler:
                with self.profiler:
                    model_config = self._train(df, **kwargs)
            else:
                model_config = self._train(df, **kwargs)
        except Exception as e:
            self.release_accelerator_lock()
            if is_cuda_error(e):
                get_logger().warning(
                    "encountered CUDA error with train params: %s", kwargs
                )
                restart_worker(e)
            elif accelerator != "cpu":
                # fallback to train on CPU
                kwargs["accelerator"] = "cpu"
                cpu_cores = self.configure_torch()
                kwargs["devices"] = 1
                self.model_args = kwargs
                start_time = time.time()
                # model_config = asyncio.wait_for(
                #     loop.run_in_executor(None, self._train, df, **kwargs), None
                # )
                # model_config = loop.run_in_executor(
                #     None, functools.partial(self._train, df, **kwargs)
                # ).result()
                if self.profiler:
                    with self.profiler:
                        model_config = self._train(df, **kwargs)
                else:
                    model_config = self._train(df, **kwargs)
            else:
                get_logger().warning("encountered error with train params: %s", kwargs)
                raise e
        finally:
            self.release_accelerator_lock()

        fit_time = time.time() - start_time
        metrics = self._get_metrics(**model_config)
        metrics["device"] = "CPU" if kwargs["accelerator"] == "cpu" else "GPU:auto"
        metrics["machine"] = socket.gethostname()
        metrics["cpu_cores"] = cpu_cores
        metrics["fit_time"] = fit_time
        metrics["model_config"] = model_config

        self._export_profiles()

        return metrics

    def _export_profiles(self):
        if not self.profiler:
            return

        self.locks["profiler"].release()
        task_key = get_worker().get_current_task()

        # Retrieve the profiling statistics sort by CPU
        profiler_result = self.profiler.key_averages(group_by_stack_n=5).table(
            sort_by="self_cpu_time_total",  # Sort by total CPU time
            row_limit=20,  # Limit the number of rows in the table (remove or adjust as needed)
        )
        output_file = os.path.join(self.profile_folder, f"{task_key}_cpu.txt")
        # Save the table to a file
        with open(output_file, "w") as f:
            f.write(profiler_result)

        if self.profile_memory:
            # sort by memory usage
            profiler_result = self.profiler.key_averages(group_by_stack_n=5).table(
                sort_by="self_cpu_memory_usage",  # Sort by total CPU time
                row_limit=20,  # Limit the number of rows in the table (remove or adjust as needed)
            )
            output_file = os.path.join(self.profile_folder, f"{task_key}_memory.txt")
            # Save the table to a file
            with open(output_file, "w") as f:
                f.write(profiler_result)

        # Optionally, save as Chrome trace file
        trace_file = os.path.join(self.profile_folder, f"{task_key}_trace.txt")
        self.profiler.export_chrome_trace(trace_file)

        stacks_file = os.path.join(self.profile_folder, f"{task_key}_stacks.txt")
        self.profiler.export_stacks(stacks_file)

    @abstractmethod
    def _build_model(self, df: pd.DataFrame, **kwargs: Any):
        """
        Build the model and instantiates self.model, self.nf
        """
        pass

    @abstractmethod
    def trainable_on_cpu(self, **kwargs: Any) -> bool:
        """
        Indicates whether the model with the given parameter can be trained on CPU if the accelerator is busy.
        """
        pass

    @abstractmethod
    def torch_num_threads(self) -> int:
        """
        The optimum number of threads set by torch in CPU mode
        via `torch.set_num_threads(x)`

        Returns:
            int: the number of threads
        """
        pass

    @abstractmethod
    def _train(self, df: pd.DataFrame, **kwargs: Any) -> dict:
        """
        Train the model with the given data.

        Args:
            df (pandas.Dataframe): Time series dataframe of the dependent variable, optionally paired with covariate(s).
            **kwargs (Any): Additional context parameters.

        Returns:
            model_config (dict): the effective model config used for training the model.
        """
        pass

    @abstractmethod
    def gpu_threshold(self) -> Tuple[float, float]:
        """
        Returns the GPU utilization and memory threshold for the model.
        Computation will be put onhold until resource usage runs below the
        specified thresholds to avoid over-parallelism.
        100 means full usage.

        Returns:
            Tuple:
                1. float, GPU utilization threshold
                2. float, GPU memory threshold
        """
        pass

    @abstractmethod
    def cpu_threshold(self) -> Tuple[float, float]:
        """
        Returns the CPU utilization and memory threshold for the model.
        Computation will be put onhold until resource usage runs below the
        specified thresholds to avoid over-parallelism.
        100 means full usage.

        Returns:
            Tuple:
                1. float, CPU utilization threshold
                2. float, CPU memory threshold
        """
        pass

    @abstractmethod
    def baseline_params(self) -> dict:
        """
        Get the baseline parameters for the model.

        Returns:
            dict: A dictionary containing the baseline parameters for the model.
        """
        pass

    @abstractmethod
    def trim_forecast(self, forecast: pd.DataFrame) -> pd.DataFrame:
        """
        Select forecast dataframe columns to align with "forecast_params" table before persistence.

        Args:
            forecast (pandas.DataFrame): The forecast dataframe to be trimmed.

        Returns:
            pandas.DataFrame: The trimmed forecast dataframe.
        """
        pass

    @abstractmethod
    def _predict(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """
        Make predictions based on the given time series dataframe and parameters.

        Args:
            df (pandas.DataFrame): Time series dataframe of the dependent variable, optionally paired with covariate(s).
            **kwargs (Any): Additional context parameters.

        Returns:
            pandas.DataFrame: The predictions dataframe. It must contain these columns
                - ds: The date of the prediction
                - y: The historical value. For future time horizons, it can be np.nan
                - yhat_n: The predicted value
        """
        pass

    def predict(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """
        Make predictions based on the given time series dataframe and parameters.

        Args:
            df (pandas.DataFrame): Time series dataframe of the dependent variable, optionally paired with covariate(s).
            **kwargs (Any): Additional context parameters.

        Returns:
            pandas.DataFrame: The predictions dataframe. It must contain these columns
                - ds: The date of the prediction
                - y: The historical value. For future time horizons, it can be np.nan
                - yhat_n: The predicted value
        """
        if self.model_args["accelerator"] == "gpu":
            self.release_accelerator_lock()
            while True:
                if self._lock_accelerator("gpu") == "gpu":
                    break
                else:
                    threading.Event().wait(0.5)
        df = df.copy()
        df.insert(0, "unique_id", "0")
        try:
            forecast = self._predict(df, **kwargs)
            # convert np.float32 type columns in forecast dataframe to native float,
            # to avoid `psycopg2.ProgrammingError: can't adapt type 'numpy.float32'`
            for col in forecast.select_dtypes(include=[np.float32]).columns:
                forecast[col] = forecast[col].astype(float)
        finally:
            self.release_accelerator_lock()
            # torch.cuda.empty_cache()

        return forecast

    def _neural_impute(self, df):
        random_seed = self.model_args["random_seed"]
        na_col = df.columns[1]
        df.rename(columns={na_col: "y"}, inplace=True)

        seed_logger = logging.getLogger("lightning_fabric.utilities.seed")
        from lightning_utilities.core.rank_zero import log as rank_zero_logger

        # rank_zero_logger = logging.getLogger("lightning_utilities.core.rank_zero")
        orig_seed_log_level = seed_logger.getEffectiveLevel()
        orig_log_level = rank_zero_logger.getEffectiveLevel()
        seed_logger.setLevel(logging.FATAL)
        rank_zero_logger.setLevel(logging.FATAL)

        np_random_seed(random_seed)
        set_log_level("ERROR")
        # optimize_torch()

        na_positions = df.isna()
        df_nona = df.dropna()
        scaler = StandardScaler()
        scaler.fit(df_nona.iloc[:, 1:])
        df_filled = df.ffill().bfill()
        df.iloc[:, 1:] = scaler.transform(df_filled.iloc[:, 1:])
        df[na_positions] = np.nan

        # accelerator = self._lock_accelerator("auto")
        # self.release_accelerator_lock(self.device_lock_release_delay)
        # accelerator = accelerator if accelerator == "gpu" else None

        # gu, _ = gpu_util()
        # accelerator = "cpu" if gu > 0 else "gpu"
        accelerator = "cpu"

        try:
            m = NeuralProphet(
                accelerator=accelerator,
                collect_metrics=False,
                trainer_config={
                    "enable_checkpointing": False,
                    "logger": False,
                    # "enable_progress_bar": False,
                    # "callbacks": [],
                },
            )
            m.fit(
                df,
                progress=None,
                checkpointing=False,
                # minimal=True,
                # metrics=None,
                trainer_config={
                    "enable_checkpointing": False,
                    "logger": False,
                    # "enable_progress_bar": False,
                    # "callbacks": []
                },
            )
        except Exception as e:
            if accelerator != "cpu":
                m = NeuralProphet(
                    accelerator="cpu",
                    collect_metrics=False,
                    trainer_config={
                        "enable_checkpointing": False,
                        "logger": False,
                        # "enable_progress_bar": False,
                        # "callbacks": [],
                    },
                )
                m.fit(
                    df,
                    progress=None,
                    checkpointing=False,
                    # minimal=True,
                    # metrics=None,
                    trainer_config={
                        "enable_checkpointing": False,
                        "logger": False,
                        # "enable_progress_bar": False,
                        # "callbacks": [],
                    },
                )
            else:
                raise e

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            forecast = m.predict(df)

        seed_logger.setLevel(orig_seed_log_level)
        rank_zero_logger.setLevel(orig_log_level)

        forecast = forecast[["ds", "yhat1"]]
        forecast["ds"] = forecast["ds"].dt.date
        forecast["yhat1"] = forecast["yhat1"].astype(float)
        forecast.rename(columns={"yhat1": na_col}, inplace=True)
        forecast.iloc[:, 1:] = scaler.inverse_transform(forecast.iloc[:, 1:])

        return forecast

    def impute(self, df, **kwargs):
        self.model_args = kwargs
        df_na = df.iloc[:, 1:].isna()

        if not df_na.any().any():
            return df, None

        na_counts = df_na.sum()
        na_cols = na_counts[na_counts > 0].index.tolist()
        na_row_indices = df[df.iloc[:, 1:].isna().any(axis=1)].index

        results = []
        for na_col in na_cols:
            df_na = df[["ds", na_col]].copy()
            if df[na_col].isnull().all():  # all rows are null
                df_na["ds"] = df_na["ds"].dt.date
                results.append(df_na)
            else:
                results.append(self._neural_impute(df_na))

        imputed_df = results[0]
        for result in results[1:]:
            imputed_df = imputed_df.merge(result, on="ds", how="left")

        for na_col in na_cols:
            # df[na_col].fillna(imputed_df[na_col], inplace=True)
            df.fillna({na_col: imputed_df[na_col]}, inplace=True)

        # Select imputed rows only
        imputed_df = imputed_df.loc[na_row_indices].copy()

        return df, imputed_df

    @abstractmethod
    def search_space(self, **kwargs: Any) -> str:
        """
        The hyper-parameter search space for optimization.

        Args:
            **kwargs (Any): Additional context parameters.

        Returns:
            str: The hyper-parameter search space in a format suitable for the optimization library and executable by eval().
        """
        pass

    @abstractmethod
    def accept_missing_data(self) -> bool:
        """
        Check if the model can handle missing data in the input.

        Returns:
            bool: True if the model can handle missing data, False otherwise.
        """
        pass
