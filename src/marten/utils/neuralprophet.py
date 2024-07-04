import torch
import re
import warnings
import traceback
import math
import pandas as pd

from tenacity import (
    stop_after_attempt,
    wait_exponential,
    Retrying,
    retry_if_exception,
    RetryError,
)
from dask.distributed import get_worker
from neuralprophet import NeuralProphet, set_random_seed, set_log_level

from marten.utils.logger import get_logger
from marten.utils.trainer import should_retry, log_retry, log_train_args


LOSS_CAP = 99.99

def layer_spec_to_list(spec):
    if spec is None:
        return []

    w, d = spec[0], spec[1]
    return [w] * d

def select_topk_features(df, ranked_features, k):
    """
    process df (dataframe): keep only the 'ds', 'y' columns, and columns with names 
    in top k elements in the ranked_features list.
    """
    top_k_features = ranked_features[:int(k)]
    columns_to_keep = ['ds', 'y'] + top_k_features
    return df[columns_to_keep]

def select_device(accelerator, util_threshold=80, vram_threshold=80):
    return (
        "gpu"
        if accelerator
        and torch.cuda.utilization() < util_threshold
        and torch.cuda.memory_usage() < vram_threshold
        else None
    )

def set_yhat_n(df):
    # Extract column names
    columns = df.columns

    # Filter columns that start with "yhat"
    yhat_columns = [col for col in columns if col.startswith('yhat')]

    # Sort columns based on the numerical part in ascending order
    yhat_columns_sorted = sorted(yhat_columns, key=lambda x: int(re.search(r'\d+', x).group()))
    # Sort columns based on the numerical part in descending order
    # yhat_columns_sorted = sorted(yhat_columns, key=lambda x: int(re.search(r'\d+', x).group()), reverse=True)

    # Initialize yhat_n with the values from the smallest yhat column
    df['yhat_n'] = df[yhat_columns_sorted[0]]
    # Initialize yhat_n with the values from the largest yhat column
    # df["yhat_n"] = df[yhat_columns_sorted[0]]

    # Iterate over the remaining yhat columns and fill in null/NA values in yhat_n
    for col in yhat_columns_sorted[1:]:
        df["yhat_n"] = df["yhat_n"].fillna(df[col])

def set_forecast_columns(forecast):
    # List of columns to keep
    columns_to_keep = ["ds", "y", "trend", "season_yearly"]

    # Add columns that match the pattern 'yhat'
    columns_to_keep += [
        col
        for col in forecast.columns
        if col.startswith("yhat") and "%" not in col
    ]

    # Remove columns not in the list of columns to keep
    forecast.drop(
        columns=[col for col in forecast.columns if col not in columns_to_keep],
        inplace=True,
    )

    set_yhat_n(forecast)


def sanitize_loss(value):
    global LOSS_CAP
    return (
        LOSS_CAP
        if math.isnan(value) or math.isinf(value) or abs(value) > LOSS_CAP
        else value
    )


class NPPredictor:

    @staticmethod
    def isBaseline(params):
        return (
            params["batch_size"] is None
            and params["n_lags"] == 0
            and params["yearly_seasonality"] == "auto"
            and params["ar_layers"] == []
            and params["lagged_reg_layers"] == []
        )

    @staticmethod
    def _try_fitting(
        df,
        epochs=None,
        random_seed=7,
        early_stopping=True,
        country=None,
        validate=True,
        **kwargs,
    ):
        set_log_level("ERROR")
        set_random_seed(random_seed)

        # m = NeuralProphet(trainer_config=_trainer_config(), **kwargs)
        m = NeuralProphet(**kwargs)
        covars = [col for col in df.columns if col not in ("ds", "y")]
        m.add_lagged_regressor(covars)
        if country is not None:
            m.add_country_holidays(country_name=country)
        try:
            if validate:
                train_df, test_df = m.split_df(
                    df,
                    valid_p=1.0 / 10,
                    freq="B",
                )
                metrics = m.fit(
                    train_df,
                    validation_df=test_df,
                    progress=None,
                    epochs=epochs,
                    early_stopping=early_stopping,
                    freq="B",
                    checkpointing=False,
                )
            else:
                metrics = m.fit(
                    df,
                    progress=None,
                    epochs=epochs,
                    early_stopping=early_stopping,
                    freq="B",
                    checkpointing=False,
                )
            return m, metrics
        except ValueError as e:
            # check if the message `Inputs/targets with missing values detected` was inside the error
            if "Inputs/targets with missing values detected" in str(e):
                # count how many 'nan' values in the `covars` columns respectively
                nan_counts = df[covars].isna().sum().to_dict()
                raise ValueError(
                    f"Skipping: too much missing values in the covariates: {nan_counts}"
                ) from e
            else:
                raise e

    @staticmethod
    def _train(
        df,
        epochs=None,
        random_seed=7,
        early_stopping=True,
        country=None,
        validate=True,
        **kwargs,
    ):
        worker = get_worker()
        logger, args = worker.logger, worker.args

        covars = [col for col in df.columns if col not in ("ds", "y")]
        wait_gpu = getattr(args, "wait_gpu", False) and (
            len(covars) >= args.wait_gpu
            or (
                "ar_layers" in kwargs
                and len(kwargs["ar_layers"]) > 0
                and kwargs["ar_layers"][0] >= 512
            )
            or (
                "lagged_reg_layers" in kwargs
                and len(kwargs["lagged_reg_layers"]) > 0
                and kwargs["lagged_reg_layers"][0] >= 512
            )
        )

        if getattr(args, "log_train_args", False):
            log_train_args(
                df, epochs, random_seed, early_stopping, country, validate, **kwargs
            )

        def _train_with_cpu():
            logger.debug("modifying **kwargs to train with cpu: %s", kwargs)
            if "accelerator" in kwargs and kwargs["accelerator"] in ["gpu", "auto"]:
                del kwargs["accelerator"]
            logger.debug("**kwargs after modification: %s", kwargs)
            m, metrics = NPPredictor._try_fitting(
                df, epochs, random_seed, early_stopping, country, validate, **kwargs
            )
            return m, metrics

        with warnings.catch_warnings():
            # suppress swarming warning:
            # WARNING - (py.warnings._showwarnmsg) -
            # ....../.pyenv/versions/3.12.2/envs/venv_3.12.2/lib/python3.12/site-packages/neuralprophet/df_utils.py:1152:
            # FutureWarning: Series.view is deprecated and will be removed in a future version. Use ``astype`` as an alternative to change the dtype.
            # converted_ds = pd.to_datetime(ds_col, utc=True).view(dtype=np.int64)
            warnings.simplefilter("ignore", FutureWarning)
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                if (
                    select_device(
                        "accelerator" in kwargs,
                        getattr(args, "gpu_util_threshold", None),
                        getattr(args, "gpu_ram_threshold", None),
                    )
                    is None
                    and not wait_gpu
                ):
                    return _train_with_cpu()

                for attempt in Retrying(
                    stop=stop_after_attempt(5 if wait_gpu else 1),
                    wait=wait_exponential(multiplier=1, max=10),
                    retry=retry_if_exception(should_retry),
                    before_sleep=log_retry,
                ):
                    with attempt:
                        m, metrics = NPPredictor._try_fitting(
                            df,
                            epochs,
                            random_seed,
                            early_stopping,
                            country,
                            validate,
                            **kwargs,
                        )
                        return m, metrics
            except RetryError as e:
                full_traceback = traceback.format_exc()
                if "OutOfMemoryError" in str(e) or "out of memory" in full_traceback:
                    # final attempt to train on CPU
                    # remove `accelerator` parameter from **kwargs
                    get_logger().warning("falling back to CPU due to OutOfMemoryError")
                    return _train_with_cpu()
                else:
                    get_logger().warning(f"falling back to CPU due to RetryError: {str(e)}")
                    return _train_with_cpu()

    @staticmethod
    def predict(m, df, random_seed, future_steps):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            # WARNING - (py.warnings._showwarnmsg) - .../python3.12/site-packages/neuralprophet/data/process.py:127:
            # PerformanceWarning: DataFrame is highly fragmented.
            # This is usually the result of calling `frame.insert` many times,
            # which has poor performance.
            # Consider joining all columns at once using pd.concat(axis=1) instead.
            # To get a de-fragmented frame, use `newframe = frame.copy()`
            # df_forecast[name] = yhat

            set_log_level("ERROR")
            set_random_seed(random_seed)

            future = m.make_future_dataframe(
                df, n_historic_predictions=True, periods=future_steps
            )
            forecast = m.predict(future)

        set_forecast_columns(forecast)
        return forecast

    @staticmethod
    def train(
        df,
        params,
        holiday_region,
        accelerator,
        early_stopping,
        random_seed,
        validate,
        epochs,
    ):
        if "ar_layers" not in params:
            params["ar_layers"] = layer_spec_to_list(params["ar_layer_spec"])
            params.pop("ar_layer_spec")
        if "lagged_reg_layers" not in params:
            params["lagged_reg_layers"] = layer_spec_to_list(
                params["lagged_reg_layer_spec"]
            )
            params.pop("lagged_reg_layer_spec")

        try:
            m, metrics = NPPredictor._train(
                df,
                epochs=epochs,
                random_seed=random_seed,
                early_stopping=early_stopping,
                weekly_seasonality=False,
                daily_seasonality=False,
                impute_missing=True,
                accelerator=accelerator,
                validate=validate,
                country=holiday_region,
                changepoints_range=1.0,
                **params,
            )
        except ValueError as e:
            get_logger().warning(str(e))
            return None
        except Exception as e:
            get_logger().exception(e)
            get_logger().error("params: %s", params)
            return None

        last_metric = metrics.iloc[-1]

        # Suppress the SettingWithCopyWarning
        pd.options.mode.chained_assignment = None

        last_metric["Loss_val"] = sanitize_loss(last_metric["Loss_val"])
        last_metric["MAE_val"] = sanitize_loss(last_metric["MAE_val"])
        last_metric["RMSE_val"] = sanitize_loss(last_metric["RMSE_val"])
        last_metric["MAE"] = sanitize_loss(last_metric["MAE"])
        last_metric["RMSE"] = sanitize_loss(last_metric["RMSE"])
        last_metric["Loss"] = sanitize_loss(last_metric["Loss"])

        covars = [col for col in df.columns if col not in ("ds", "y")]
        get_logger().debug("params:%s\n#covars:%s\n%s", params, len(covars), last_metric)

        return m, last_metric
