import time
import pandas as pd
import json
import hashlib
import warnings
import traceback
import math
import torch
import os

OPENBLAS_NUM_THREADS = 1
os.environ["OPENBLAS_NUM_THREADS"] = f"{OPENBLAS_NUM_THREADS}"

from datetime import datetime, timedelta

from sqlalchemy import text

from neuralprophet import NeuralProphet, set_random_seed, set_log_level
from neuralprophet.hdays_utils import get_country_holidays

from dask.distributed import get_worker, worker_client

from tenacity import (
    stop_after_attempt,
    wait_exponential,
    Retrying,
    retry_if_exception,
    RetryError,
)

from marten.utils.worker import await_futures
from marten.utils.holidays import get_holiday_region
from marten.utils.logger import get_logger
from marten.utils.pl import GlobalProgressBar
from marten.utils.neuralprophet import (
    select_topk_features,
    layer_spec_to_list,
    select_device,
    set_forecast_columns,
)


from types import SimpleNamespace

LOSS_CAP = 99.99


def merge_covar_df(
    anchor_symbol, anchor_df, cov_table, cov_symbol, feature, min_date, alchemyEngine
):

    if anchor_symbol == cov_symbol:
        if feature == "y":
            # no covariate is needed. this is a baseline metric
            merged_df = anchor_df[["ds", "y"]]
        else:
            # using endogenous features as covariate
            merged_df = anchor_df[["ds", "y", feature]]

        return merged_df

    cov_symbol_sanitized = f"{feature}_{cov_symbol}"
    cutoff_date = anchor_df["ds"].max().strftime("%Y-%m-%d")

    match cov_table:
        case "bond_metrics_em" | "bond_metrics_em_view" | "currency_boc_safe_view":
            query = f"""
                select date ds, {feature} "{cov_symbol_sanitized}"
                from {cov_table}
                where date >= %(min_date)s
                and date <= %(cutoff_date)s
                and {feature} is not null
                and {feature} <> 'nan'
                order by date
            """
            params = {
                "min_date": min_date,
                "cutoff_date": cutoff_date,
            }
        case _:
            query = f"""
                select date ds, {feature} "{cov_symbol_sanitized}"
                from {cov_table}
                where symbol = %(cov_symbol)s
                and date >= %(min_date)s
                and date <= %(cutoff_date)s
                and {feature} is not null
                and {feature} <> 'nan'
                order by date
            """
            params = {
                "cov_symbol": cov_symbol,
                "min_date": min_date,
                "cutoff_date": cutoff_date,
            }

    with alchemyEngine.connect() as conn:
        cov_symbol_df = pd.read_sql(query, conn, params=params, parse_dates=["ds"])

    if cov_symbol_df.empty:
        return None

    merged_df = pd.merge(anchor_df, cov_symbol_df, on="ds", how="left")

    return merged_df


def fit_with_covar(
    anchor_symbol,
    anchor_df,
    cov_table,
    cov_symbol,
    min_date,
    random_seed,
    feature,
    accelerator,
    early_stopping,
    infer_holiday,
):
    # Local import of get_worker to avoid circular import issue?
    worker = get_worker()
    alchemyEngine, logger, args = worker.alchemyEngine, worker.logger, worker.args

    merged_df = merge_covar_df(
        anchor_symbol,
        anchor_df,
        cov_table,
        cov_symbol,
        feature,
        min_date,
        alchemyEngine,
    )

    covar_col = feature if feature in merged_df.columns else f"{feature}_{cov_symbol}"
    nan_count = int(merged_df[covar_col].isna().sum())
    if nan_count >= len(merged_df) * 0.5:
        logger.info("too much missing values in %s: %s, skipping", covar_col, nan_count)
        return None

    if merged_df is None:
        # FIXME: sometimes merged_df is None even if there's data in table
        logger.info(
            "skipping covariate: %s, %s, %s, %s",
            cov_table,
            cov_symbol,
            feature,
            min_date,
        )
        return None

    start_time = time.time()
    metrics = None
    region = None
    if infer_holiday:
        region = get_holiday_region(alchemyEngine, anchor_symbol)
    try:
        _, metrics = train(
            df=merged_df,
            epochs=args.epochs,
            random_seed=random_seed,
            early_stopping=early_stopping,
            batch_size=None,
            yearly_seasonality="auto",
            weekly_seasonality=False,
            daily_seasonality=False,
            impute_missing=True,
            accelerator=accelerator,
            validate=True,
            country=region,
            changepoints_range=1.0,
        )
    except ValueError as e:
        logger.warning(str(e))
        return None
    except Exception as e:
        logger.exception(e)
        return None
    fit_time = time.time() - start_time
    # extract the last row of output, add symbol column, and consolidate to another dataframe
    last_row = metrics.iloc[[-1]]
    # get the row count in merged_df as timesteps
    timesteps = len(merged_df)
    # get merged_df's right-most column's nan count.
    ts_cutoff_date = merged_df["ds"].max().strftime("%Y-%m-%d")
    with alchemyEngine.begin() as conn:
        save_covar_metrics(
            anchor_symbol,
            cov_table,
            cov_symbol,
            feature,
            last_row,
            fit_time,
            timesteps,
            nan_count,
            ts_cutoff_date,
            conn,
        )
    return last_row


def save_covar_metrics(
    anchor_symbol,
    cov_table,
    cov_symbol,
    feature,
    cov_metrics,
    fit_time,
    timesteps,
    nan_count,
    ts_cutoff_date,
    conn,
):
    # Inserting DataFrame into the database table
    for _, row in cov_metrics.iterrows():
        epochs = row["epoch"] + 1
        conn.execute(
            text(
                """
                    INSERT INTO neuralprophet_corel 
                    (symbol, cov_table, cov_symbol, feature, mae_val, 
                    rmse_val, loss_val, fit_time, timesteps, nan_count, 
                    ts_date, epochs, mae, rmse, loss) 
                    VALUES (:symbol, :cov_table, :cov_symbol, :feature, :mae_val, 
                    :rmse_val, :loss_val, :fit_time, :timesteps, :nan_count, 
                    :ts_date, :epochs, :mae, :rmse, :loss) 
                    ON CONFLICT (symbol, cov_symbol, feature, cov_table, ts_date) 
                    DO UPDATE SET 
                        mae_val = EXCLUDED.mae_val, 
                        rmse_val = EXCLUDED.rmse_val, 
                        loss_val = EXCLUDED.loss_val,
                        fit_time = EXCLUDED.fit_time,
                        timesteps = EXCLUDED.timesteps,
                        nan_count = EXCLUDED.nan_count,
                        epochs = EXCLUDED.epochs,
                        mae = EXCLUDED.mae, 
                        rmse = EXCLUDED.rmse, 
                        loss = EXCLUDED.loss
                """
            ),
            {
                "symbol": anchor_symbol,
                "cov_table": cov_table,
                "cov_symbol": cov_symbol,
                "feature": feature,
                "ts_date": ts_cutoff_date,
                "mae_val": row["MAE_val"],
                "rmse_val": row["RMSE_val"],
                "loss_val": row["Loss_val"],
                "fit_time": (
                    (str(fit_time) + " seconds") if fit_time is not None else None
                ),
                "timesteps": timesteps,
                "nan_count": nan_count,
                "epochs": epochs,
                "mae": row["MAE"],
                "rmse": row["RMSE"],
                "loss": row["Loss"],
            },
        )


def log_retry(retry_state):
    if retry_state.outcome.failed:
        exception = retry_state.outcome.exception()
        get_logger().warning(
            f"Retrying, attempt {retry_state.attempt_number} after exception: {exception}"
        )


def _trainer_config():
    # TODO: can we use custom trainer config to specify gpu and device?
    config = {
        "callbacks": [
            GlobalProgressBar(False, False),  # suppress/disable progress bar display
        ]
    }
    return config


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


def should_retry(exception):
    return isinstance(exception, torch.cuda.OutOfMemoryError) or (
        "out of memory" in str(exception)
    )


def log_train_args(df, *args, **kwargs):
    worker = get_worker()
    logger = worker.logger
    logger.info(
        (
            "Model training arguments:\n"
            "Dataframe %s:\n%s\n"
            "Positional arguments:%s\n"
            "Keyword arguments:%s"
        ),
        df.shape,
        df.describe().to_string(),
        args,
        kwargs,
    )


def train(
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
        or ("ar_layers" in kwargs and kwargs["ar_layers"][0] >= 512)
        or ("lagged_reg_layers" in kwargs and kwargs["lagged_reg_layers"][0] >= 512)
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
        m, metrics = _try_fitting(
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
                    m, metrics = _try_fitting(
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


def reg_search_params(params):
    if "ar_reg" in params:
        params["ar_reg"] = round(params["ar_reg"], 5)
    if "seasonality_reg" in params:
        params["seasonality_reg"] = round(params["seasonality_reg"], 5)
    if "trend_reg" in params:
        params["trend_reg"] = round(params["trend_reg"], 5)


def validate_hyperparams(args, df, ranked_features, covar_set_id, hps_id, params):
    reg_params = params.copy()
    reg_search_params(reg_params)
    loss_val = log_metrics_for_hyper_params(
        args.symbol,
        df,
        reg_params,
        args.epochs,
        args.random_seed,
        select_device(
            args.accelerator,
            getattr(args, "gpu_util_threshold", None),
            getattr(args, "gpu_ram_threshold", None),
        ),
        covar_set_id,
        hps_id,
        args.early_stopping,
        args.infer_holiday,
        ranked_features,
    )
    return (params, loss_val)


def log_metrics_for_hyper_params(
    anchor_symbol,
    df,
    params,
    epochs,
    random_seed,
    accelerator,
    covar_set_id,
    hps_id,
    early_stopping,
    infer_holiday,
    ranked_features,
):
    worker = get_worker()
    alchemyEngine, logger, args = worker.alchemyEngine, worker.logger, worker.args

    # to support distributed processing, we try to insert a new record (with primary keys only)
    # into hps_metrics first. If we hit duplicated key error, return None.
    # Otherwise we could proceed further code execution.
    param_str = json.dumps(params, sort_keys=True)
    hpid = hashlib.md5(param_str.encode("utf-8")).hexdigest()
    if not new_metric_keys(
        anchor_symbol, hpid, param_str, covar_set_id, hps_id, alchemyEngine
    ):
        logger.debug("Skip re-entry for %s: %s", anchor_symbol, param_str)
        with alchemyEngine.connect() as conn:
            result = conn.execute(
                text(
                    """
                        select loss_val 
                        from hps_metrics
                        where model = :model
                        and anchor_symbol = :anchor_symbol
                        and hpid = :hpid
                        and hps_id = :hps_id
                    """
                ),
                {
                    "model": "NeuralProphet",
                    "anchor_symbol": anchor_symbol,
                    "hpid": hpid,
                    "hps_id": hps_id,
                },
            )
            row = result.fetchone()
            if row is not None:
                loss_val = row[0]
            return loss_val

    if "ar_layers" not in params:
        params["ar_layers"] = layer_spec_to_list(params["ar_layer_spec"])
        params.pop("ar_layer_spec")
    if "lagged_reg_layers" not in params:
        params["lagged_reg_layers"] = layer_spec_to_list(
            params["lagged_reg_layer_spec"]
        )
        params.pop("lagged_reg_layer_spec")

    topk_covar = None
    if "topk_covar" in params:
        topk_covar = params["topk_covar"]
        params.pop("topk_covar")

    start_time = time.time()
    metrics = None
    region = None

    if infer_holiday:
        region = get_holiday_region(alchemyEngine, anchor_symbol)

    if topk_covar is not None:
        df = select_topk_features(df, ranked_features, topk_covar)

    try:
        _, metrics = train(
            df,
            epochs=epochs,
            random_seed=random_seed,
            early_stopping=early_stopping,
            weekly_seasonality=False,
            daily_seasonality=False,
            impute_missing=True,
            accelerator=accelerator,
            validate=True,
            country=region,
            changepoints_range=1.0,
            **params,
        )
    except ValueError as e:
        logger.warning(str(e))
        return None
    except Exception as e:
        logger.exception(e)
        logger.error("params: %s", params)
        return None

    fit_time = time.time() - start_time
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
    logger.debug("params:%s\n#covars:%s\n%s", params, len(covars), last_metric)

    update_metrics_table(
        alchemyEngine,
        params,
        anchor_symbol,
        hpid,
        last_metric["epoch"] + 1,
        last_metric,
        fit_time,
        covar_set_id,
        topk_covar,
    )

    return last_metric["Loss_val"]


def sanitize_all_loss(df):
    columns_to_sanitize = ["Loss_val", "MAE_val", "RMSE_val", "MAE", "RMSE", "Loss"]
    for column in columns_to_sanitize:
        if column in df.columns:
            df.loc[df.index[-1], column] = sanitize_loss(df.loc[df.index[-1], column])


def sanitize_loss(value):
    global LOSS_CAP
    return (
        LOSS_CAP
        if math.isnan(value) or math.isinf(value) or abs(value) > LOSS_CAP
        else value
    )


def update_metrics_table(
    alchemyEngine,
    params,
    anchor_symbol,
    hpid,
    epochs,
    last_metric,
    fit_time,
    covar_set_id,
    topk_covar,
):
    def action():
        with alchemyEngine.begin() as conn:
            tag = None
            if (
                params["batch_size"] is None
                and params["n_lags"] == 0
                and params["yearly_seasonality"] == "auto"
                and params["ar_layers"] == []
                and params["lagged_reg_layers"] == []
            ):
                tag = (
                    "baseline,univariate"
                    if covar_set_id == 0
                    else "baseline,multivariate"
                )
            conn.execute(
                text(
                    """
                    UPDATE hps_metrics
                    SET 
                        mae_val = :mae_val, 
                        rmse_val = :rmse_val, 
                        loss_val = :loss_val, 
                        mae = :mae,
                        rmse = :rmse,
                        loss = :loss,
                        fit_time = :fit_time,
                        epochs = :epochs,
                        tag = :tag,
                        sub_topk = :sub_topk
                    WHERE
                        model = :model
                        AND anchor_symbol = :anchor_symbol
                        AND hpid = :hpid
                        AND covar_set_id = :covar_set_id
                """
                ),
                {
                    "model": "NeuralProphet",
                    "anchor_symbol": anchor_symbol,
                    "hpid": hpid,
                    "covar_set_id": covar_set_id,
                    "tag": tag,
                    "mae_val": last_metric["MAE_val"],
                    "rmse_val": last_metric["RMSE_val"],
                    "loss_val": last_metric["Loss_val"],
                    "mae": last_metric["MAE"],
                    "rmse": last_metric["RMSE"],
                    "loss": last_metric["Loss"],
                    "fit_time": (
                        (str(fit_time) + " seconds") if fit_time is not None else None
                    ),
                    "epochs": epochs,
                    "sub_topk": topk_covar,
                },
            )

    for attempt in Retrying(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=5)
    ):
        with attempt:
            action()


def new_metric_keys(
    anchor_symbol, hpid, hyper_params, covar_set_id, hps_id, alchemyEngine
):
    def action():
        try:
            with alchemyEngine.begin() as conn:
                conn.execute(
                    text(
                        """
                        INSERT INTO hps_metrics (model, anchor_symbol, hpid, hyper_params, covar_set_id, hps_id) 
                        VALUES (:model, :anchor_symbol, :hpid, :hyper_params, :covar_set_id, :hps_id)
                        """
                    ),
                    {
                        "model": "NeuralProphet",
                        "anchor_symbol": anchor_symbol,
                        "hpid": hpid,
                        "hps_id": hps_id,
                        "hyper_params": hyper_params,
                        "covar_set_id": covar_set_id,
                    },
                )
                return True
        except Exception as e:
            if "duplicate key value violates unique constraint" in str(e):
                return False
            else:
                raise e

    for attempt in Retrying(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=5)
    ):
        with attempt:
            return action()


def get_topk_foundation_settings(
    alchemyEngine, symbol, hps_id, topk, ts_date, nan_limit
):
    # worker = get_worker()
    # alchemyEngine = worker.alchemyEngine

    query = """
        WITH univ_baseline_nc as (
            select cov_table, cov_symbol, feature, symbol
            from neuralprophet_corel
            where 
                symbol = %(symbol)s
                and cov_symbol = symbol
                and feature = 'y'
                and ts_date = %(ts_date)s
        ),
        univ_baseline as (
            select
                hm.hyper_params, hm.mae, hm.rmse, hm.loss, hm.mae_val, 
                hm.rmse_val, hm.loss_val, hm.hpid, hm.epochs, hm.sub_topk,
                hm.covar_set_id, hm.anchor_symbol symbol,
                nc.cov_table, nc.cov_symbol, nc.feature
            from hps_metrics hm
            inner join univ_baseline_nc nc
                on hm.anchor_symbol = nc.symbol
            where hm.anchor_symbol = %(symbol)s 
            and hm.hps_id = %(hps_id)s
            and hm.covar_set_id = 0
        ),
        top_by_loss_val as (
            SELECT 
                ub.hyper_params, nc.mae, nc.rmse, nc.loss, nc.mae_val,
                nc.rmse_val, nc.loss_val, ub.hpid, nc.epochs, 1, 
                0, nc.symbol, nc.cov_table, nc.cov_symbol, nc.feature
            FROM neuralprophet_corel nc
            INNER JOIN
                univ_baseline ub
            ON nc.symbol = ub.symbol
            where 
                nc.ts_date = %(ts_date)s
                and nc.loss_val < ub.loss_val
                and nc.nan_count < %(nan_limit)s
            order by nc.loss_val
            limit %(limit)s
        ),
        top_by_loss as (
            SELECT 
                ub.hyper_params, nc.mae, nc.rmse, nc.loss, nc.mae_val,
                nc.rmse_val, nc.loss_val, ub.hpid, nc.epochs, 1, 
                0, nc.symbol, nc.cov_table, nc.cov_symbol, nc.feature
            FROM neuralprophet_corel nc
            INNER JOIN
                univ_baseline ub
            ON nc.symbol = ub.symbol
            where 
                nc.ts_date = %(ts_date)s
                and nc.loss_val < ub.loss_val
                and nc.nan_count <= %(nan_limit)s
            order by nc.loss
            limit %(limit)s
        )
        SELECT DISTINCT *
        FROM univ_baseline
        UNION
        SELECT DISTINCT *
        FROM top_by_loss_val
        UNION
        SELECT DISTINCT *
        FROM top_by_loss
    """
    params = {
        "symbol": symbol,
        "hps_id": hps_id,
        "limit": topk,
        "ts_date": ts_date,
        "nan_limit": nan_limit,
    }

    df = pd.read_sql(query, alchemyEngine, params=params)
    df.drop("symbol", axis=1, inplace=True)

    return df


def get_topk_prediction_settings(alchemyEngine, symbol, hps_id, topk):
    # worker = get_worker()
    # alchemyEngine = worker.alchemyEngine

    query = """
        WITH baseline as (
            select loss_val
            from hps_metrics
            where anchor_symbol = %(symbol)s 
            and hps_id = %(hps_id)s 
            and covar_set_id = 0
        ),
        top_by_loss_val AS (
            SELECT 
                hyper_params, mae, rmse, loss, mae_val, 
                rmse_val, loss_val, hpid, epochs, sub_topk,
                covar_set_id
            FROM hps_metrics
            WHERE anchor_symbol = %(symbol)s
            AND hps_id = %(hps_id)s 
            and loss_val < (select loss_val from baseline)
            ORDER BY loss_val
            LIMIT %(limit)s
        ),
        top_by_loss AS (
            SELECT 
                hyper_params, mae, rmse, loss, mae_val, 
                rmse_val, loss_val, hpid, epochs, sub_topk,
                covar_set_id
            FROM hps_metrics
            WHERE anchor_symbol = %(symbol)s
            AND hps_id = %(hps_id)s 
            and loss_val < (select loss_val from baseline)
            ORDER BY loss
            LIMIT %(limit)s
        )
        SELECT DISTINCT *
        FROM top_by_loss_val
        UNION
        SELECT DISTINCT *
        FROM top_by_loss
    """
    params = {
        "symbol": symbol,
        "hps_id": hps_id,
        "limit": topk,
    }

    return pd.read_sql(query, alchemyEngine, params=params)


def get_best_prediction_setting(alchemyEngine, logger, symbol, df, topk, nth_top):
    # find the model setting with optimum performance, including univariate default setting.
    from marten.models.hp_search import (
        default_params,
        augment_anchor_df_with_covars,
    )
    from marten.utils.neuralprophet import layer_spec_to_list

    query = """
        select
            *
        from
        (
            (
                select
                    null cov_table,
                    null cov_symbol,
                    null feature,
                    hm.hyper_params,
                    hm.loss_val,
                    hm.covar_set_id,
                    null nan_count,
                    hm.sub_topk,
                    hs.ts_date
                from
                    hps_metrics hm
                join hps_sessions hs
                on hm.hps_id = hs.id
                    and hm.anchor_symbol = hs.symbol
                where
                    hm.anchor_symbol = %(symbol)s
                order by
                    hs.ts_date desc,
                    hm.loss_val asc
                limit %(limit)s
            )
        union all 
            (
                select
                    cov_table,
                    cov_symbol,
                    feature,
                    null hyper_params,
                    loss_val,
                    null covar_set_id,
                    nan_count,
                    null sub_topk,
                    ts_date
                from
                    neuralprophet_corel nc
                where
                    symbol = %(symbol)s
                order by
                    ts_date desc,
                    loss_val asc
                limit %(limit)s
            )
        )
        order by
            ts_date desc,
            loss_val asc
        offset %(offset)s rows
        fetch first 1 row only
    """
    params = {
        "symbol": symbol,
        "limit": topk,
        "offset": nth_top - 1,
    }

    with alchemyEngine.connect() as conn:
        best_setting = pd.read_sql(query, conn, params=params)

    hyperparams = None
    covar_set_id = None
    new_df = None
    if best_setting.empty:
        logger.warn(
            (
                "No hyper-parameter search or covariate metrics for %s. "
                "Proceeding univariate prediction with default hyper-parameters. "
                "This might not be the most accurate prediction."
            ),
            symbol,
        )
        hyperparams = default_params
        new_df = df[["ds", "y"]]
    elif best_setting["hyper_params"].iloc[0] is not None:
        best_setting = best_setting.iloc[0]
        covar_set_id = int(best_setting["covar_set_id"])
        logger.info(
            (
                "%s - found best setting in hps_metrics:\n"
                "loss_val: %s\n"
                "covar_set_id: %s\n"
                "sub_topk: %s\n"
                "ts_date: %s"
            ),
            symbol,
            best_setting["loss_val"],
            covar_set_id,
            best_setting["sub_topk"],
            best_setting["ts_date"],
        )

        hyperparams = best_setting["hyper_params"]
        new_df, _, ranked_features = augment_anchor_df_with_covars(
            df,
            SimpleNamespace(covar_set_id=covar_set_id, symbol=symbol),
            alchemyEngine,
            logger,
            datetime.now().strftime("%Y%m%d"),
        )

        if best_setting["sub_topk"] is not None:
            new_df = select_topk_features(
                new_df, ranked_features, best_setting["sub_topk"]
            )
    else:
        best_setting = best_setting.iloc[0]
        logger.info(
            (
                "%s - found best setting in neuralprophet_corel:\n"
                "loss_val: %s\n"
                "cov_table: %s\n"
                "cov_symbol: %s\n"
                "feature: %s\n"
                "nan_count: %s\n",
                "ts_date: %s",
            ),
            symbol,
            best_setting["loss_val"],
            best_setting["cov_table"],
            best_setting["cov_symbol"],
            best_setting["feature"],
            best_setting["nan_count"],
            best_setting["ts_date"],
        )
        hyperparams = default_params
        new_df = merge_covar_df(
            symbol,
            df[["ds", "y"]],
            best_setting["cov_table"],
            best_setting["cov_symbol"],
            best_setting["feature"],
            df["ds"].min().strftime("%Y-%m-%d"),
            alchemyEngine,
        )

    if "ar_layers" not in hyperparams:
        hyperparams["ar_layers"] = layer_spec_to_list(hyperparams["ar_layer_spec"])
        hyperparams.pop("ar_layer_spec")
    if "lagged_reg_layers" not in hyperparams:
        hyperparams["lagged_reg_layers"] = layer_spec_to_list(
            hyperparams["lagged_reg_layer_spec"]
        )
        hyperparams.pop("lagged_reg_layer_spec")
    if "topk_covar" in hyperparams:
        hyperparams.pop("topk_covar")

    return new_df, hyperparams, covar_set_id


# Function to check if a date is a holiday and return the holiday name
def check_holiday(date, country_holidays):
    return country_holidays.get(date) if date in country_holidays else None


def trim_forecasts_by_dates(forecast):
    future_date = forecast.iloc[-1]["ds"]
    days_count = (future_date - datetime.now()).days + 1
    start_date = future_date - timedelta(days=days_count * 2)
    forecast = forecast[forecast["ds"] >= start_date]

    return forecast


def save_forecast_snapshot(
    alchemyEngine,
    symbol,
    hyper_params,
    covar_set_id,
    metrics,
    metrics_final,
    forecast,
    proc_time,
    region,
    random_seed,
    future_steps,
    n_covars,
    cutoff_date,
    group_id,
    hpid,
    avg_loss,
    covar,
):
    metric = metrics.iloc[-1]
    metric_final = metrics_final.iloc[-1]

    mean_diff = (forecast["yhat_n"] - forecast["y"]).mean()
    std_diff = (forecast["yhat_n"] - forecast["y"]).std()

    with alchemyEngine.begin() as conn:
        result = conn.execute(
            text(
                """
                insert
                    into
                    predict_snapshots
                    (
                        model,symbol,hyper_params,covar_set_id,mae_val,rmse_val,loss_val,mae,rmse,loss,
                        predict_diff_mean,predict_diff_stddev,epochs,proc_time,mae_final,rmse_final,loss_final,
                        region,random_seed,future_steps,n_covars,cutoff_date,group_id,hpid,avg_loss,covar
                    )
                values(
                    :model,:symbol,:hyper_params,:covar_set_id,:mae_val,:rmse_val,:loss_val,
                    :mae,:rmse,:loss,:predict_diff_mean,:predict_diff_stddev,:epochs,:proc_time,
                    :mae_final,:rmse_final,:loss_final,:region,:random_seed,:future_steps,:n_covars,
                    :cutoff_date,:group_id,:hpid,:avg_loss,:covar
                ) RETURNING id
                """
            ),
            {
                "model": "NeuralProphet",
                "symbol": symbol,
                "hyper_params": hyper_params,
                "covar_set_id": covar_set_id,
                "mae_val": metric["MAE_val"],
                "rmse_val": metric["RMSE_val"],
                "loss_val": metric["Loss_val"],
                "mae": metric["MAE"],
                "rmse": metric["RMSE"],
                "loss": metric["Loss"],
                "predict_diff_mean": mean_diff,
                "predict_diff_stddev": std_diff,
                "epochs": metric["epoch"] + 1,
                "proc_time": (
                    f"{str(proc_time)} seconds" if proc_time is not None else None
                ),
                "mae_final": metric_final["MAE"],
                "rmse_final": metric_final["RMSE"],
                "loss_final": metric_final["Loss"],
                "region": region,
                "random_seed": random_seed,
                "future_steps": future_steps,
                "n_covars": n_covars,
                "cutoff_date": cutoff_date,
                "group_id": group_id,
                "hpid": hpid,
                "avg_loss": avg_loss,
                "covar": covar,
            },
        )
        snapshot_id = result.fetchone()[0]

        ## save to ft_yearly_params table
        forecast = trim_forecasts_by_dates(forecast)

        country_holidays = get_country_holidays(region)

        forecast_params = forecast[["ds", "trend", "season_yearly", "yhat_n"]].copy()

        forecast_params.rename(columns={"ds": "date"}, inplace=True)
        forecast_params.loc[:, "symbol"] = symbol
        forecast_params.loc[:, "snapshot_id"] = snapshot_id
        forecast_params.loc[:, "holiday"] = forecast_params["date"].apply(
            lambda x: check_holiday(x, country_holidays)
        )
        forecast_params.to_sql("forecast_params", conn, if_exists="append", index=False)

    return snapshot_id, len(forecast_params)


def train_predict(
    df,
    epochs,
    random_seed,
    early_stopping,
    country,
    validate,
    future_steps,
    **kwargs,
):

    m, metrics = train(
        df=df,
        country=country,
        epochs=epochs,
        random_seed=random_seed,
        early_stopping=early_stopping,
        validate=validate,
        n_forecasts=future_steps,
        **kwargs,
    )

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
    return forecast, metrics


def calc_final_forecast(forecast, mode):
    match mode:
        case "multiplicative":
            forecast["forecast"] = forecast["trend"] * forecast["season_yearly"]
        case _:  # "additive" | "auto" | None
            forecast["forecast"] = forecast["trend"] + forecast["season_yearly"]
    return forecast


def measure_needed_mem(df, hp):
    df_shape = df.shape
    dim = df_shape[0] * df_shape[1]

    ar_layer = getattr(hp, "ar_layer", None)
    lagged_reg_layer = getattr(hp, "lagged_reg_layer", None)

    if ar_layer is None or len(ar_layer) == 0:
        al_dim = 1
    else:
        al_dim = len(ar_layer) * ar_layer[0]

    if lagged_reg_layer is None or len(lagged_reg_layer) == 0:
        lrl_dim = 1
    else:
        lrl_dim = len(lagged_reg_layer) * lagged_reg_layer[0]

    return dim * al_dim * lrl_dim / 10.5


def forecast(symbol, df, ranked_features, hps_metric, region, cutoff_date, group_id):
    worker = get_worker()
    alchemyEngine, logger, args = worker.alchemyEngine, worker.logger, worker.args

    hyperparams = hps_metric["hyper_params"]
    covar_set_id = int(hps_metric["covar_set_id"])
    hp_str = json.dumps(hyperparams, sort_keys=True)
    logger.info(
        (
            "%s - forecasting with setting:\n"
            "hpid: %s\n"
            "loss: %s\n"
            "loss_val: %s\n"
            "covar_set_id: %s\n"
            "sub_topk: %s\n"
            "cutoff_date: %s\n"
            "hyper-params: %s"
        ),
        symbol,
        hps_metric["hpid"],
        hps_metric["loss"],
        hps_metric["loss_val"],
        covar_set_id,
        hps_metric["sub_topk"],
        cutoff_date,
        hp_str,
    )

    if covar_set_id == 0:
        new_df = merge_covar_df(
            symbol,
            df[["ds", "y"]],
            hps_metric["cov_table"],
            hps_metric["cov_symbol"],
            hps_metric["feature"],
            df["ds"].min().strftime("%Y-%m-%d"),
            alchemyEngine,
        )
    else:
        # new_df, _, ranked_features = augment_anchor_df_with_covars(
        #     df,
        #     SimpleNamespace(covar_set_id=covar_set_id, symbol=symbol),
        #     alchemyEngine,
        #     logger,
        #     cutoff_date,
        # )
        if hps_metric["sub_topk"] is None:
            new_df = df
        else:
            new_df = select_topk_features(
                df, ranked_features, int(hps_metric["sub_topk"])
            )

    logger.info(
        "dataframe augmented with covar_set_id %s: %s", covar_set_id, new_df.shape
    )

    if "ar_layers" not in hyperparams:
        hyperparams["ar_layers"] = layer_spec_to_list(hyperparams["ar_layer_spec"])
        hyperparams.pop("ar_layer_spec")
    if "lagged_reg_layers" not in hyperparams:
        hyperparams["lagged_reg_layers"] = layer_spec_to_list(
            hyperparams["lagged_reg_layer_spec"]
        )
        hyperparams.pop("lagged_reg_layer_spec")
    if "topk_covar" in hyperparams:
        hyperparams.pop("topk_covar")

    start_time = time.time()
    with worker_client() as client:
        df_future = client.scatter(new_df)
        futures = []
        # train with validation
        futures.append(
            client.submit(
                train,
                df=df_future,
                epochs=args.epochs,
                random_seed=args.random_seed,
                early_stopping=args.early_stopping,
                weekly_seasonality=False,
                daily_seasonality=False,
                impute_missing=True,
                accelerator="gpu" if args.accelerator else None,
                validate=True,
                country=region,
                changepoints_range=1.0,
                **hyperparams,
            )
        )
        # train with full dataset without validation split
        # worker_mem_needed = measure_needed_mem(new_df, hyperparams)
        futures.append(
            client.submit(
                train_predict,
                df=df_future,
                country=region,
                epochs=args.epochs,
                random_seed=args.random_seed,
                early_stopping=args.early_stopping,
                weekly_seasonality=False,
                daily_seasonality=False,
                impute_missing=True,
                validate=False,
                future_steps=args.future_steps,
                changepoints_range=1.0,
                accelerator="gpu" if args.accelerator else None,
                **hyperparams,
                # resources={"MEMORY": worker_mem_needed},
            )
        )
        results = client.gather(futures)

    proc_time = time.time() - start_time
    metrics = results[0][1]
    forecast, metrics_final = results[1][0], results[1][1]

    logger.debug("forecast columns: %s", forecast.columns)
    # NOTE get forecasted y values directly from `forecast` output
    # calc_final_forecast(forecast,
    # hyperparams["seasonality_mode"] if "seasonality_mode" in hyperparams else None)

    # metrics.loc[metrics.index[-1]] is used to get a view of the last row,
    # and modifications to this view will be reflected in the original DataFrame.
    sanitize_all_loss(metrics)
    sanitize_all_loss(metrics_final)

    # metrics.iloc[-1]["Loss_val"] = sanitize_loss(metrics.iloc[-1]["Loss_val"])
    # metrics_final.iloc[-1]["Loss"] = sanitize_loss(metrics_final.iloc[-1]["Loss"])
    avg_loss = round(
        (metrics.iloc[-1]["Loss_val"] + metrics_final.iloc[-1]["Loss"]) / 2.0, 5
    )

    covar_columns = [col for col in new_df.columns if col not in ("ds", "y")]
    n_covars = len(covar_columns)

    snapshot_id, n_yearly_seasonality = save_forecast_snapshot(
        alchemyEngine,
        symbol,
        hp_str,
        covar_set_id,
        metrics,
        metrics_final,
        forecast,
        proc_time,
        region,
        args.random_seed,
        args.future_steps,
        n_covars,
        cutoff_date,
        group_id,
        hps_metric["hpid"],
        avg_loss,
        covar_columns[0] if n_covars == 1 else None,
    )

    logger.info(
        "%s - estimated %s yearly-seasonality coefficients. snapshot_id: %s, averaged loss: %s",
        symbol,
        n_yearly_seasonality,
        snapshot_id,
        avg_loss,
    )

    return snapshot_id, forecast, avg_loss


def get_prediction_group_id(alchemyEngine):
    with alchemyEngine.begin() as conn:
        return conn.execute(text("SELECT nextval('prediction_group_id_seq')")).scalar()


def ensemble_topk_prediction(
    client,
    symbol,
    timestep_limit,
    random_seed,
    future_steps,
    topk,
    hps_id,
    cutoff_date,
    ranked_features_future,
    df,
    df_future,
    alchemyEngine,
    logger,
    args,
):
    # worker = get_worker()
    # alchemyEngine, logger, args = worker.alchemyEngine, worker.logger, worker.args

    # NOTE: we don't specify the cutoff_date to load TS, such that
    # we can predict based off latest historical data. Some model HPs
    # may not generalize well to unseen data.
    # df, _ = load_anchor_ts(symbol, timestep_limit, alchemyEngine)

    region = get_holiday_region(alchemyEngine, symbol)
    logger.info("%s - inferred holiday region: %s", symbol, region)

    s1 = get_topk_prediction_settings(alchemyEngine, symbol, hps_id, topk)
    # get univariate and 2*topk 2-pair covariate settings
    nan_threshold = round(len(df) * args.nan_limit, 0)
    s2 = get_topk_foundation_settings(
        alchemyEngine, symbol, hps_id, topk, cutoff_date, nan_threshold
    )
    settings = pd.concat([s1, s2], axis=0, ignore_index=True)

    group_id = get_prediction_group_id(alchemyEngine)
    logger.info("prediction settings loaded: %s, group id: %s", len(settings), group_id)

    # scale-in to preserve more memory for prediction
    logger.info("Scaling dask cluster to %s", args.min_worker)
    client.cluster.scale(args.min_worker)

    futures = []
    for _, row in settings.iterrows():
        futures.append(
            client.submit(
                forecast,
                symbol,
                df_future,
                ranked_features_future,
                row,
                region,
                cutoff_date,
                group_id,
            )
        )

    results = client.gather(futures)

    # each row of results is a tuple consisting snapshot_id, forecast, avg_loss
    # select the topk rows with lowest agg_loss value
    topk_results = sorted(results, key=lambda x: x[2])[:topk]
    snapshot_ids = [t[0] for t in topk_results]
    forecasts = [t[1] for t in topk_results]
    avg_loss = [t[2] for t in topk_results]

    sum_loss = sum(avg_loss)
    weights = [round((1.0 - (loss / sum_loss)) / (topk - 1.0), 5) for loss in avg_loss]

    save_ensemble_snapshot(
        alchemyEngine,
        symbol,
        forecasts,
        avg_loss,
        weights,
        region,
        snapshot_ids,
        random_seed,
        future_steps,
        group_id,
    )


def predict_best(
    symbol,
    early_stopping,
    timestep_limit,
    epochs,
    random_seed,
    future_steps,
    topk,
    accelerator,
):
    from marten.models.hp_search import (
        load_anchor_ts,
    )

    worker = get_worker()
    alchemyEngine, logger = worker.alchemyEngine, worker.logger

    df, _ = load_anchor_ts(symbol, timestep_limit, alchemyEngine)

    region = get_holiday_region(alchemyEngine, symbol)
    logger.info("%s - inferred holiday region: %s", symbol, region)

    # use the best performing setting to fit and then predict
    # start_time = time.time()
    futures = []
    dfs = []
    params_list = []
    covarset_id_list = []
    merged_df = None
    results = None
    with worker_client() as client:
        for i in range(1, topk + 1):
            # TODO improve performance by multi-processing the `get_best_prediction_setting()`
            merged_df, params, covar_set_id = get_best_prediction_setting(
                alchemyEngine, logger, symbol, df, topk, i
            )
            dfs.append(merged_df)
            params_json = json.dumps(params, sort_keys=True)
            params_list.append(params_json)
            covarset_id_list.append(covar_set_id)
            logger.info(
                "%s - using hyper-parameters for top-%s setting:\n%s",
                symbol,
                i,
                params_json,
            )
            df_future = client.scatter(merged_df)
            # train with validation
            futures.append(
                client.submit(
                    train,
                    df=df_future,
                    country=region,
                    epochs=epochs,
                    random_seed=random_seed,
                    early_stopping=early_stopping,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    impute_missing=True,
                    validate=True,
                    changepoints_range=1.0,
                    accelerator=accelerator,
                    **params,
                )
            )
            # train with full dataset without validation split
            futures.append(
                client.submit(
                    train_predict,
                    df=df_future,
                    country=region,
                    epochs=epochs,
                    random_seed=random_seed,
                    early_stopping=early_stopping,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    impute_missing=True,
                    validate=False,
                    future_steps=future_steps,
                    changepoints_range=1.0,
                    accelerator=accelerator,
                    **params,
                )
            )
        results = client.gather(futures)

    top_forecasts = []
    agg_loss = []
    snapshot_ids = []

    for i, df, params_json, cid in zip(
        range(0, topk * 2, 2), dfs, params_list, covarset_id_list
    ):
        metrics = results[i][1]
        forecast, metrics_final = results[i + 1][0], results[i + 1][1]
        agg_loss.append(metrics.iloc[-1]["Loss_val"] + metrics_final.iloc[-1]["Loss"])

        # fit_time = time.time() - start_time
        # TODO do we want to record the accurate fit time?
        fit_time = None

        top_forecasts.append(forecast)

        n_covars = len([col for col in df.columns if col not in ("ds", "y")])
        cutoff_date = df["ds"].max().strftime("%Y-%m-%d")

        # save the snapshot and yearly seasonality coefficients to tables.
        snapshot_id, n_yearly_seasonality = save_forecast_snapshot(
            alchemyEngine,
            symbol,
            params_json,
            cid,
            metrics,
            metrics_final,
            forecast,
            fit_time,
            region,
            random_seed,
            future_steps,
            n_covars,
            cutoff_date,
        )
        snapshot_ids.append(snapshot_id)

        logger.info(
            "%s - estimated %s yearly-seasonality coefficients. snapshot_id: %s",
            symbol,
            n_yearly_seasonality,
            snapshot_id,
        )

    if topk <= 1:
        return

    sum_loss = sum(agg_loss)
    weights = [(1.0 - (loss / sum_loss)) / (topk - 1.0) for loss in agg_loss]

    save_ensemble_snapshot(
        alchemyEngine,
        symbol,
        top_forecasts,
        weights,
        region,
        snapshot_ids,
        random_seed,
        future_steps,
    )


def save_ensemble_snapshot(
    alchemyEngine,
    symbol,
    top_forecasts,
    avg_loss,
    weights,
    region,
    snapshot_ids,
    random_seed,
    future_steps,
    group_id,
):
    ens_df = None
    country_holidays = get_country_holidays(region)
    hyper_params = json.dumps(
        [
            {"snapshot_id": sid, "weight": w, "avg_loss": loss}
            for sid, w, loss in zip(snapshot_ids, weights, avg_loss)
        ],
        sort_keys=True,
    )

    for df, w in zip(top_forecasts, weights):
        df = trim_forecasts_by_dates(df)

        df = df[["ds", "yhat_n"]]
        df.rename(columns={"ds": "date"}, inplace=True)
        df["yhat_n"] = df["yhat_n"] * w

        if ens_df is None:
            ens_df = df
            ens_df.loc[:, "symbol"] = symbol
            ens_df.loc[:, "holiday"] = ens_df["date"].apply(
                lambda x: check_holiday(x, country_holidays)
            )
            ens_df.set_index("date", inplace=True)
        else:
            df.set_index("date", inplace=True)
            ens_df["yhat_n"] += df["yhat_n"]

    ens_df.reset_index(inplace=True)

    with alchemyEngine.begin() as conn:
        result = conn.execute(
            text(
                """
                insert
                    into
                    predict_snapshots
                    (
                        model,symbol,hyper_params,
                        region,random_seed,future_steps,
                        group_id
                    )
                values(
                    :model,:symbol,:hyper_params,
                    :region,:random_seed,:future_steps,
                    :group_id
                ) RETURNING id
                """
            ),
            {
                "model": "NeuralProphet_Ensemble",
                "symbol": symbol,
                "hyper_params": hyper_params,
                "region": region,
                "random_seed": random_seed,
                "future_steps": future_steps,
                "group_id": group_id,
            },
        )
        snapshot_id = result.fetchone()[0]
        ens_df.loc[:, "snapshot_id"] = snapshot_id
        ens_df.to_sql("forecast_params", conn, if_exists="append", index=False)


def init_hps(hps, symbol, args, client, alchemyEngine, logger):
    # worker = get_worker()
    # alchemyEngine, logger = worker.alchemyEngine, worker.logger

    args.symbol = symbol
    args.hps_only = False
    args.covar_only = False
    args.infer_holiday = True
    args.method = "fast_bayesopt"

    hps.logger = logger
    hps.alchemyEngine = alchemyEngine
    hps.args = args

    hps.client = client

    return args


def count_topk_hp(alchemyEngine, hps_id, base_loss):
    with alchemyEngine.connect() as conn:
        result = conn.execute(
            text(
                """
                    select count(*)
                    from hps_metrics
                    where 
                        hps_id = :hps_id
                        and loss_val <= :base_loss
                """
            ),
            {
                "hps_id": hps_id,
                "base_loss": base_loss,
            },
        )
        return result.fetchone()[0]


def fast_bayesopt(
    client, alchemyEngine, logger, df, covar_set_id, hps_id, ranked_features, base_loss, args
):
    # worker = get_worker()
    # logger = worker.logger

    from scipy.stats import uniform
    from marten.models.hp_search import (
        _cleanup_stale_keys,
        _search_space,
        _bayesopt_run,
        update_hps_sessions,
    )

    _cleanup_stale_keys()

    space_str = _search_space(args.max_covars)

    # Convert args to a dictionary, excluding non-serializable items
    args_dict = {k: v for k, v in vars(args).items() if not callable(v)}
    args_json = json.dumps(args_dict, sort_keys=True)
    update_hps_sessions(hps_id, "fast_bayesopt", args_json, space_str, covar_set_id)

    n_jobs = args.batch_size

    # split large iterations into smaller runs to avoid OOM / memory leak
    for i in range(args.max_itr):
        logger.info("running bayesopt mini-iteration %s/%s", i + 1, args.max_itr)
        min_loss = _bayesopt_run(
            df,
            n_jobs,
            covar_set_id,
            hps_id,
            ranked_features,
            eval(space_str, {"uniform": uniform}),
            args,
            args.mini_itr,
            args.resume or i > 0,
        )

        if min_loss is None or min_loss > base_loss:
            continue

        topk_count = count_topk_hp(alchemyEngine, hps_id, base_loss)

        if topk_count >= args.topk:
            logger.info(
                "Found %s HP with Loss_val less than %s. Best score: %s, stopping bayesopt.",
                topk_count,
                base_loss,
                min_loss,
            )
            return topk_count
        else:
            logger.info(
                "Found %s HP with Loss_val less than %s. Best score: %s",
                topk_count,
                base_loss,
                min_loss,
            )


def update_covar_set_id(alchemyEngine, hps_id, covar_set_id):
    sql = """
        update hps_sessions
        set covar_set_id = :covar_set_id
        where id = :hps_id
    """
    with alchemyEngine.begin() as conn:
        conn.execute(text(sql), {"hps_id": hps_id, "covar_set_id": covar_set_id})


def _univariate_default_hp(client, anchor_df, args, hps_id):
    from marten.models.hp_search import default_params

    df = anchor_df[["ds", "y"]]
    return client.submit(
        log_metrics_for_hyper_params,
        args.symbol,
        df,
        default_params,
        args.epochs,
        args.random_seed,
        select_device(
            args.accelerator,
            getattr(args, "gpu_util_threshold", None),
            getattr(args, "gpu_ram_threshold", None),
        ),
        0,
        hps_id,
        args.early_stopping,
        args.infer_holiday,
        None,
    ).result()


def min_covar_loss_val(alchemyEngine, symbol, ts_date):
    with alchemyEngine.connect() as conn:
        result = conn.execute(
            text(
                """
                    select min(loss_val)
                    from neuralprophet_corel
                    where symbol = :symbol 
                        and ts_date = :ts_date
                """
            ),
            {
                "symbol": symbol,
                "ts_date": ts_date,
            },
        )
        return result.fetchone()[0]


def covars_and_search(client, symbol, alchemyEngine, logger, args):
    global LOSS_CAP

    import marten.models.hp_search as hps
    from marten.models.hp_search import (
        _get_cutoff_date,
        load_anchor_ts,
        get_hps_session,
        prep_covar_baseline_metrics,
        augment_anchor_df_with_covars,
    )

    # worker = get_worker()
    # alchemyEngine, logger, args = worker.alchemyEngine, worker.logger, worker.args

    args = init_hps(hps, symbol, args, client, alchemyEngine, logger)
    cutoff_date = _get_cutoff_date(args)
    anchor_df, anchor_table = load_anchor_ts(
        args.symbol, args.timestep_limit, alchemyEngine, cutoff_date
    )
    cutoff_date = anchor_df["ds"].max().strftime("%Y-%m-%d")

    hps_id, covar_set_id = get_hps_session(
        args.symbol, cutoff_date, args.resume, len(anchor_df)
    )
    args.covar_set_id = covar_set_id
    logger.info(
        "HPS session ID: %s, Cutoff date: %s, CovarSet ID: %s, Anchor Table: %s",
        hps_id,
        cutoff_date,
        covar_set_id,
        anchor_table,
    )

    univ_loss = _univariate_default_hp(client, anchor_df, args, hps_id)

    min_covar_loss = min_covar_loss_val(alchemyEngine, symbol, cutoff_date)
    min_covar_loss = min_covar_loss if min_covar_loss is not None else LOSS_CAP

    base_loss = min(float(univ_loss) * args.loss_quantile, min_covar_loss)

    # if in resume mode, check if the topk HP is present, and further check if prediction is already conducted.
    topk_count = count_topk_hp(alchemyEngine, hps_id, base_loss)
    if args.resume and topk_count >= args.topk:
        logger.info(
            "Found %s HP with Loss_val less than %s in HP search history already. Skipping covariate and HP search.",
            topk_count,
            base_loss,
        )
        df, covar_set_id, ranked_features = augment_anchor_df_with_covars(
            anchor_df, args, alchemyEngine, logger, cutoff_date
        )
        df_future = client.scatter(df)
        ranked_features_future = client.scatter(ranked_features)
        return hps_id, cutoff_date, ranked_features_future, df, df_future
    else:
        logger.info(
            "Found %s HP with Loss_val less than %s in HP search history. The process will be continued.",
            topk_count,
            base_loss,
        )

    # scale up the cluster to args.max_worker
    logger.info("Scaling dask cluster to %s", args.max_worker)
    client.cluster.scale(args.max_worker)

    # run covariate loss calculation in batch
    logger.info("Starting covariate loss calculation")
    t1_start = time.time()
    prep_covar_baseline_metrics(anchor_df, anchor_table, args)
    logger.debug("waiting dask futures: %s", len(hps.futures))
    await_futures(hps.futures, hard_wait=True)

    logger.info(
        "%s covariate baseline metric computation completed. Time taken: %s seconds",
        args.symbol,
        round(time.time() - t1_start, 3),
    )

    min_covar_loss = min_covar_loss_val(alchemyEngine, symbol, cutoff_date)
    min_covar_loss = min_covar_loss if min_covar_loss is not None else LOSS_CAP
    base_loss = min(base_loss, min_covar_loss)

    # run HP search using Bayeopt and check whether needed HP(s) are found
    logger.info(
        "Starting Bayesian optimization search for hyper-parameters. Loss_val threshold: %s",
        round(base_loss, 5),
    )

    # scale-in to preserve more memory for hps
    worker_size = max(args.min_worker, round(args.max_worker * 0.8))
    logger.info("Scaling down dask cluster to %s", worker_size)
    client.cluster.scale(worker_size)

    # NOTE: if data is scattered before scale-down, the error will be thrown:
    # Removing worker 'tcp://<worker IP & port>' caused the cluster to lose scattered data, which can't be recovered
    df, covar_set_id, ranked_features = augment_anchor_df_with_covars(
        anchor_df, args, alchemyEngine, logger, cutoff_date
    )
    # df_future = client.scatter(df)
    # ranked_features_future = client.scatter(ranked_features)

    t2_start = time.time()

    update_covar_set_id(alchemyEngine, hps_id, covar_set_id)

    fast_bayesopt(
        client,
        alchemyEngine,
        logger,
        df,
        covar_set_id,
        hps_id,
        ranked_features,
        base_loss,
        args,
    )
    logger.info(
        "%s hyper-parameter search completed. Time taken: %s seconds",
        args.symbol,
        round(time.time() - t2_start, 3),
    )
    await_futures(hps.futures)

    return hps_id, cutoff_date, ranked_features_future, df, df_future


def predict_adhoc(symbol, args):
    worker = get_worker()
    logger = worker.logger

    with worker_client() as client:
        hps_id, cutoff_date, ranked_features_future, df, df_future = covars_and_search(
            client, symbol
        )

    # predict
    logger.info("Starting adhoc prediction")
    t3_start = time.time()
    ensemble_topk_prediction(
        symbol,
        args.timestep_limit,
        args.random_seed,
        args.future_steps,
        args.topk,
        hps_id,
        cutoff_date,
        ranked_features_future,
        df,
        df_future,
    )
    logger.info(
        "%s prediction completed. Time taken: %s seconds",
        symbol,
        round(time.time() - t3_start, 3),
    )
