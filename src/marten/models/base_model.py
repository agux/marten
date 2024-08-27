from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import Any, Tuple
import time
import socket

import pandas as pd

import torch

from dask.distributed import get_worker, worker_client, Lock, get_client

from marten.utils.logger import get_logger
from marten.utils.trainer import optimize_torch, is_cuda_error
from marten.utils.worker import release_lock, wait_gpu, wait_cpu, restart_worker, workload_stage

class BaseModel(ABC):

    def __init__(self) -> None:
        super().__init__()
        self.device = None
        self.model_args = None
        self.accelerator_lock = None

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

    def _check_cpu(self) -> bool:
        match workload_stage():
            case "finishing":
                return False  # stick to GPU and avoid straggler last-task
            case "starting":
                return True
            case _:
                return self.trainable_on_cpu(**self.model_args)

    def _lock_accelerator(self, accelerator) -> Lock:
        gpu_lock_key = f"""{socket.gethostname()}::GPU-auto"""
        cpu_lock_key = f"""{socket.gethostname()}::CPU"""
        resource_wait_time = 5  # seconds, waiting for compute resource
        lock_wait_time = 2
        gpu_ut, gpu_rt = self.gpu_threshold()
        cpu_ut, cpu_rt = self.cpu_threshold()

        while True:
            lock_acquired = None
            if accelerator == "gpu":
                lock = Lock(gpu_lock_key)
                if lock.acquire(timeout=f"{lock_wait_time}s"):
                    lock_acquired = lock
                    get_logger().debug("lock acquired: %s", gpu_lock_key)
            if lock_acquired is None and self._check_cpu():
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
            elif self._check_cpu():  # CPU
                while wait_cpu(cpu_ut, cpu_rt, stop_at):
                    time.sleep(1)
            else:
                release_lock(lock_acquired, 0)
                continue

            if time.time() <= stop_at:
                break

            release_lock(lock_acquired, 0)

        self.accelerator_lock = lock_acquired

        return lock_acquired

    def _select_accelerator(self, accelerator) -> str:
        accelerator = accelerator.lower()
        # if accelerator == "cpu":
        #     return accelerator
        if accelerator in ("gpu", "auto") and not torch.cuda.is_available():
            accelerator = "cpu"

        # gpu or auto
        worker = get_worker()
        if worker is not None:
            task_key = worker.get_current_task()
            if worker.client.get_metadata([task_key, "CUDA error"], False):
                accelerator = "cpu"

        self.release_accelerator_lock()
        lock = self._lock_accelerator(accelerator)
        accelerator = accelerator if "GPU" in lock.name else "cpu"
        self.release_accelerator_lock(2 if self.is_baseline(**self.model_args) else 7)
        return accelerator

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

    def train(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """
        Select the proper accelerator and train the model with the given data.

        Args:
            df (pandas.Dataframe): Time series dataframe of the dependent variable, optionally paired with covariate(s).
            **kwargs (Any): Additional context parameters.

        Returns:
            pandas.DataFrame: The metrics of the training process. It must contain these columns
                - epochs: 0-based;
                - MAE_val: Mean Average Error for validation set;
                - RMSE_val: Root Mean Squared Error for validation set;
                - Loss_val: indicative loss value for validation set;
                - MAE: Mean Average Error for training set;
                - RMSE: Root Mean Squared Error for training set;
                - Loss: indicative loss value for training set;
                - device: e.g. GPU:0, CPU, etc.
                - machine: which machine the model is trained on. Can be obtained from socket.gethostname()
        """
        self.model_args = kwargs
        accelerator = self._select_accelerator(kwargs["accelerator"])
        kwargs["accelerator"] = accelerator
        if accelerator == "cpu":
            optimize_torch(self.torch_cpu_ratio())
            kwargs["devices"] = 1
        else:
            kwargs["devices"] = "auto"
        self.model_args = kwargs
        try:
            metrics = self._train(df, **kwargs)
        except Exception as e:
            self.release_accelerator_lock()
            if is_cuda_error(e):
                restart_worker(e)
            elif accelerator != "cpu":
                # fallback to train on CPU
                kwargs["accelerator"] = "cpu"
                optimize_torch(self.torch_cpu_ratio())
                kwargs["devices"] = 1
                self.model_args = kwargs
                metrics = self._train(df, **kwargs)
            else:
                raise e

        metrics["device"] = "CPU" if kwargs["accelerator"] == "cpu" else "GPU:auto"
        metrics['machine'] = socket.gethostname()

        return metrics

    @abstractmethod
    def trainable_on_cpu(self, **kwargs: Any) ->bool:
        """
        Indicates whether the model with the given parameter can be trained on CPU if the accelerator is busy.
        """
        pass

    @abstractmethod
    def torch_cpu_ratio(self) -> float:
        """
        Before we set the number of threads for torch in CPU mode
        via `torch.set_num_threads(x)`, we need to calculate the x number.
        Final x will take into account the ratio returned by this function,
        which represents the porportion of idle CPU cores at the time of execution.
        Returning 1 will likely cause the subsequent computation to render a 100% CPU usage.

        Returns:
            float: the ratio representing a porportion of idle CPU cores at the time of execution
        """
        pass

    @abstractmethod
    def _train(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """
        Train the model with the given data.

        Args:
            df (pandas.Dataframe): Time series dataframe of the dependent variable, optionally paired with covariate(s).
            **kwargs (Any): Additional context parameters.

        Returns:
            pandas.DataFrame: The metrics of the training process. It must contain these columns
                - epochs: 0-based;
                - MAE_val: Mean Average Error for validation set;
                - RMSE_val: Root Mean Squared Error for validation set;
                - Loss_val: indicative loss value for validation set;
                - MAE: Mean Average Error for training set;
                - RMSE: Root Mean Squared Error for training set;
                - Loss: indicative loss value for training set;
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
        pass

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
