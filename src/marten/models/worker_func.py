import time
import pandas as pd
import json
import hashlib
import traceback
import math
import os

OPENBLAS_NUM_THREADS = 1
os.environ["OPENBLAS_NUM_THREADS"] = f"{OPENBLAS_NUM_THREADS}"

import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import text
from psycopg2.extras import execute_values
from neuralprophet.event_utils import get_all_holidays
import dask
from dask.distributed import get_worker, worker_client
from types import SimpleNamespace
from tenacity import (
    stop_after_attempt,
    wait_exponential,
    Retrying,
)

from marten.data.worker_func import impute
from marten.utils.worker import await_futures
from marten.utils.holidays import get_holiday_region
from marten.utils.logger import get_logger
from marten.utils.trainer import select_device, get_accelerator_locks
from marten.utils.softs import SOFTSPredictor, baseline_config
from marten.utils.database import columns_with_prefix
from marten.utils.neuralprophet import (
    select_topk_features,
    NPPredictor,
)


LOSS_CAP = 99.99


def merge_covar_df(
    anchor_symbol, anchor_df, cov_table, cov_symbol, feature, min_date, alchemyEngine, sem
):

    if anchor_symbol == cov_symbol and not cov_table.startswith("ta_"):
        if feature == "y":
            # no covariate is needed. this is a baseline metric
            merged_df = anchor_df[["ds", "y"]]
            return merged_df
        else:
            # using endogenous features as covariate
            col_name = f"{feature}::{cov_table}::{cov_symbol}"
            if col_name in anchor_df.columns:
                merged_df = anchor_df[["ds", "y", col_name]]
                return merged_df
            elif feature in anchor_df.columns:
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
        case _ if cov_table.startswith("ta_"):  # handle technical indicators table
            column_names = columns_with_prefix(alchemyEngine, cov_table, feature)
            columns = ", ".join([f'{c} "{c}_{cov_symbol}"' for c in column_names])
            # split cov_symbol by "::" so that we get the "table" from the first element and
            # the real "cov_symbol" from the second
            cov_symbol_table, cov_symbol = cov_symbol.split("::")
            query = f"""
                select date ds, {columns}
                from {cov_table}
                where symbol = %(cov_symbol)s
                and "table" = %(table)s
                and date >= %(min_date)s
                and date <= %(cutoff_date)s
                order by date
            """
            params = {
                "cov_symbol": cov_symbol,
                "table": cov_symbol_table,
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

    if sem:
        with sem:
            with alchemyEngine.connect() as conn:
                cov_symbol_df = pd.read_sql(query, conn, params=params, parse_dates=["ds"])
    else:
        with alchemyEngine.connect() as conn:
            cov_symbol_df = pd.read_sql(query, conn, params=params, parse_dates=["ds"])

    if cov_symbol_df.empty:
        return None

    merged_df = anchor_df[["ds", "y"]].copy()
    merged_df = pd.merge(merged_df, cov_symbol_df, on="ds", how="left")

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
    sem=None,
    locks=None,
):
    # Local import of get_worker to avoid circular import issue?
    worker = get_worker()
    alchemyEngine, logger, args = worker.alchemyEngine, worker.logger, worker.args
    model = worker.model

    def _func():
        merged_df = merge_covar_df(
            anchor_symbol,
            anchor_df,
            cov_table,
            cov_symbol,
            feature,
            min_date,
            alchemyEngine,
            sem,
        )

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

        if cov_table.startswith("ta_"):
            covar_col = [
                col for col in merged_df.columns if col.startswith(f"{feature}_")
            ]
        else:
            covar_col = (
                feature if feature in merged_df.columns else f"{feature}_{cov_symbol}"
            )
        nan_count = int(merged_df[covar_col].isna().sum().sum())
        if nan_count >= merged_df.shape[0] * merged_df.shape[1] * 0.5:
            logger.info(
                "too much missing values in %s: %s, skipping", covar_col, nan_count
            )
            return None

        start_time = time.time()
        impute_df = None
        match args.model:
            case "NeuralProphet":
                region = (
                    get_holiday_region(alchemyEngine, anchor_symbol)
                    if infer_holiday
                    else None
                )
                params = {
                    "yearly_seasonality": "auto",
                }
                _, metrics = NPPredictor.train(
                    df=merged_df,
                    params=params,
                    holiday_region=region,
                    accelerator=accelerator,
                    early_stopping=early_stopping,
                    random_seed=random_seed,
                    validate=True,
                    epochs=args.epochs,
                )
            case "SOFTS":
                model_id = f"baseline_{anchor_symbol}_paired_covar"
                config = baseline_config.copy()
                config["pred_len"] = args.future_steps
                config["train_epochs"] = args.epochs
                config["use_gpu"] = accelerator == True or accelerator == "gpu"
                merged_df, impute_df = impute(merged_df, random_seed)
                m, metrics = SOFTSPredictor.train(
                    merged_df, config, model_id, random_seed, True
                )
                m.cleanup()
            case _:
                if not model.accept_missing_data():
                    df_na = merged_df.iloc[:, 1:].isna()
                    if df_na.any().any():
                        logger.info(
                            "running imputation for %s @ %s.%s",
                            cov_symbol,
                            cov_table,
                            feature,
                        )
                    merged_df, impute_df = impute(merged_df, random_seed)
                config = model.baseline_params()
                config["h"] = args.future_steps
                config["max_steps"] = args.epochs
                config["accelerator"] = (
                    accelerator
                    if isinstance(accelerator, str)
                    else "auto" if accelerator == True else "cpu"
                )
                config["validate"] = True
                config["random_seed"] = random_seed
                config["locks"] = locks
                metrics = model.train(merged_df, **config)

        fit_time = time.time() - start_time
        # get the row count in merged_df as timesteps
        timesteps = len(merged_df)
        # get merged_df's right-most column's nan count.
        ts_cutoff_date = merged_df["ds"].max().strftime("%Y-%m-%d")
        with alchemyEngine.begin() as conn:
            save_covar_metrics(
                args.model,
                anchor_symbol,
                cov_table,
                cov_symbol,
                feature,
                metrics,
                fit_time,
                timesteps,
                nan_count,
                ts_cutoff_date,
                conn,
            )
        if impute_df is not None:
            logger.info(
                "saving imputated data points for %s, %s.%s: %s",
                cov_symbol,
                cov_table,
                feature,
                impute_df.shape,
            )
            save_impute_data(
                impute_df, cov_table, cov_symbol, feature, alchemyEngine, logger
            )
        return metrics

    try:
        return _func()
    except Exception as e:
        logger.error(
            "failed to fit covar %s @ %s.%s:\n%s",
            cov_symbol,
            cov_table,
            feature,
            traceback.format_exc(),
        )
        raise e


def save_impute_data(impute_df, cov_table, cov_symbol, feature, alchemyEngine, logger):
    cov_table = cov_table[:-5] if cov_table.endswith("_view") else cov_table
    if cov_table.startswith("ta_"):
        # saving imputation for technical indicators, where multiple columns could be infolved
        df_cols = [col for col in impute_df.columns if col.startswith(f"{feature}_")]
        column_names = columns_with_prefix(alchemyEngine, cov_table, feature)
        column_names.sort(key=len, reverse=True)
        cols = []
        exclude = []
        table, symbol = cov_symbol.split("::")
        for df_col in df_cols:
            for tbl_col in column_names:
                if df_col.startswith(tbl_col + "_"):
                    cols.append(tbl_col)
                    exclude.append(f"{tbl_col} = EXCLUDED.{tbl_col}")
                    break
        sql = f"""
                INSERT INTO {cov_table}_impute (symbol, "table", date, {", ".join(cols)}) 
                VALUES %s 
                ON CONFLICT (symbol, "table", date) 
                DO UPDATE SET 
                    {", ".join(exclude)}
              """
        impute_df.insert(0, "symbol", symbol)
        impute_df.insert(1, "table", table)
    else:
        last_col = impute_df.columns[-1]
        impute_df = impute_df[["ds", last_col]]
        match cov_table:
            case "currency_boc_safe" | "bond_metrics_em":
                sql = f"""
                    INSERT INTO {cov_table}_impute (date, {feature}) 
                    VALUES %s 
                    ON CONFLICT (date) 
                    DO UPDATE SET 
                        {feature} = EXCLUDED.{feature}
                """
            case _:
                sql = f"""
                    INSERT INTO {cov_table}_impute (symbol, date, {feature}) 
                    VALUES %s 
                    ON CONFLICT (symbol, date) 
                    DO UPDATE SET 
                        {feature} = EXCLUDED.{feature}
                """
                impute_df.insert(0, "symbol", cov_symbol)

        impute_df.rename(
            columns={"ds": "date", last_col: f"{feature}"},
            inplace=True,
        )

    with alchemyEngine.begin() as conn:
        cursor = conn.connection.cursor()  # Create a cursor from the connection
        try:
            execute_values(cursor, sql, list(impute_df.to_records(index=False)))
        except Exception as e:
            logger.warning(
                "%s imputation not persisted:%s\n%s", feature, str(e), impute_df
            )
            raise e
    # conn.commit()  # Commit the transaction
    # cursor.close()  # Close the cursor


def save_covar_metrics(
    model,
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
    epochs = cov_metrics["epoch"] + 1

    if "device" in cov_metrics or "machine" in cov_metrics:
        device_info = json.dumps(
            {"device": cov_metrics["device"], "machine": cov_metrics["machine"]},
            sort_keys=True,
        )
    else:
        device_info = json.dumps({})

    params = {
        "symbol": anchor_symbol,
        "cov_table": cov_table,
        "cov_symbol": cov_symbol,
        "feature": feature,
        "ts_date": ts_cutoff_date,
        "mae_val": cov_metrics["MAE_val"],
        "rmse_val": cov_metrics["RMSE_val"],
        "loss_val": cov_metrics["Loss_val"],
        "fit_time": ((str(fit_time) + " seconds") if fit_time is not None else None),
        "timesteps": timesteps,
        "nan_count": nan_count,
        "epochs": epochs,
        "mae": cov_metrics["MAE"],
        "rmse": cov_metrics["RMSE"],
        "loss": cov_metrics["Loss"],
        "device_info": device_info,
    }
    match model:
        case "NeuralProphet":
            sql = """
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
        case _:
            sql = """
                INSERT INTO paired_correlation 
                (model, symbol, cov_table, cov_symbol, feature, mae_val, 
                rmse_val, loss_val, fit_time, timesteps, nan_count, 
                ts_date, epochs, mae, rmse, loss, device_info) 
                VALUES (:model, :symbol, :cov_table, :cov_symbol, :feature, :mae_val, 
                :rmse_val, :loss_val, :fit_time, :timesteps, :nan_count, 
                :ts_date, :epochs, :mae, :rmse, :loss, :device_info) 
                ON CONFLICT (symbol, cov_symbol, feature, cov_table, ts_date, model) 
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
                    loss = EXCLUDED.loss,
                    device_info = EXCLUDED.device_info
            """
            params["model"] = model
    conn.execute(
        text(sql),
        params,
    )


def train(
    model_name,
    df,
    epochs=None,
    random_seed=7,
    early_stopping=True,
    country=None,
    validate=True,
    save_model_file=False,
    **kwargs,
):
    worker = get_worker()
    args = worker.args
    model = worker.model

    match model_name:
        case "NeuralProphet":
            return NPPredictor.train(
                df=df,
                holiday_region=country,
                accelerator=kwargs.pop("accelerator"),
                early_stopping=early_stopping,
                random_seed=random_seed,
                validate=validate,
                epochs=epochs,
                params=kwargs,
            )
        case "SOFTS":
            kwargs["pred_len"] = args.future_steps
            kwargs["train_epochs"] = epochs
            kwargs["use_gpu"] = (
                kwargs["accelerator"] == True or kwargs["accelerator"] == "gpu"
            )
            return SOFTSPredictor.train(
                df,
                config=kwargs,
                model_id=f"""generic_model{"_validate" if validate else ""}""",
                random_seed=random_seed,
                validate=validate,
                save_model_file=save_model_file,
            )
        case _:
            config = kwargs.copy()
            config["accelerator"] = (
                kwargs["accelerator"]
                if isinstance(kwargs["accelerator"], str)
                else "auto" if kwargs["accelerator"] == True else "cpu"
            )
            config["max_steps"] = epochs
            config["h"] = args.future_steps
            metrics = model.train(
                df, random_seed=random_seed, validate=validate, **config
            )
            return (None, metrics)


def reg_search_params(params):
    if "ar_reg" in params:
        params["ar_reg"] = round(params["ar_reg"], 5)
    if "seasonality_reg" in params:
        params["seasonality_reg"] = round(params["seasonality_reg"], 5)
    if "trend_reg" in params:
        params["trend_reg"] = round(params["trend_reg"], 5)


def validate_hyperparams(args, df, covar_set_id, hps_id, params, locks):
    reg_params = params.copy()
    if args.model == "NeuralProphet":
        reg_search_params(reg_params)
    try:
        loss_val = log_metrics_for_hyper_params(
            args.symbol,
            df,
            reg_params,
            args.epochs,
            args.random_seed,
            # select_device(
            #     args.accelerator,
            #     getattr(args, "gpu_util_threshold", None),
            #     getattr(args, "gpu_ram_threshold", None),
            # ),
            args.accelerator,
            covar_set_id,
            hps_id,
            args.early_stopping,
            args.infer_holiday,
            locks,
        )
    except Exception as e:
        get_logger().error(
            "encountered error with train params: %s\n%s",
            reg_params,
            traceback.format_exc(),
        )
        raise e
    return (params, loss_val)


def get_hpid(params):
    params = params.copy()
    if "covar_dist" in params:
        # params.pop("covar_dist")
        if isinstance(params["covar_dist"], np.ndarray):
            params["covar_dist"] = params["covar_dist"].tolist()
    param_str = json.dumps(params, sort_keys=True)
    hpid = hashlib.md5(param_str.encode("utf-8")).hexdigest()
    return hpid, param_str


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
    locks=None,
):
    worker = get_worker()
    alchemyEngine, logger, args = worker.alchemyEngine, worker.logger, worker.args
    model = worker.model

    params = params.copy()

    # to support distributed processing, we try to insert a new record (with primary keys only)
    # into hps_metrics first. If we hit duplicated key error, return that validation loss.
    # Otherwise we could proceed further code execution.
    hpid, param_str = get_hpid(params)
    if not new_metric_keys(
        args.model, anchor_symbol, hpid, param_str, covar_set_id, hps_id, alchemyEngine
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
                    "model": args.model,
                    "anchor_symbol": anchor_symbol,
                    "hpid": hpid,
                    "hps_id": hps_id,
                },
            )
            row = result.fetchone()
            if row is not None:
                loss_val = row[0]
            return loss_val

    topk_covar = None
    if "topk_covar" in params:
        topk_covar = params["topk_covar"]
        params.pop("topk_covar")

    # if topk_covar is not None:
    # if "covar_dist" in params:
    # df = select_randk_covars(df, ranked_features, params["covar_dist"], topk_covar)
    # params.pop("covar_dist")
    # else:
    # df = select_topk_features(df, ranked_features, topk_covar)

    start_time = time.time()
    tag = None

    match args.model:
        case "NeuralProphet":
            region = (
                get_holiday_region(alchemyEngine, anchor_symbol)
                if infer_holiday
                else None
            )
            _, last_metric = NPPredictor.train(
                df,
                params,
                epochs=epochs,
                random_seed=random_seed,
                early_stopping=early_stopping,
                accelerator=accelerator,
                validate=True,
                holiday_region=region,
            )
            if NPPredictor.isBaseline(params):
                tag = (
                    "baseline,univariate"
                    if covar_set_id == 0
                    else "baseline,multivariate"
                )
        case "SOFTS":
            if SOFTSPredictor.isBaseline(params):
                tag = (
                    "baseline,univariate"
                    if covar_set_id == 0
                    else "baseline,multivariate"
                )
            # dataset settings
            # 'root_path': './dataset/ETT-small/',
            # 'data_path': 'ETTm1.csv',
            # 'data': 'ETTm1',
            # 'seq_len': 200,
            params["pred_len"] = args.future_steps
            # model settings
            params["train_epochs"] = epochs
            # 'd_model': 128,
            # 'd_core': 64,
            # 'd_ff': 128,
            # 'e_layers': 2,
            # 'learning_rate': 0.0003,
            # 'lradj': 'cosine',
            # 'patience': 3,
            # 'batch_size': 16,
            # 'dropout': 0.0,
            # 'activation': 'gelu',
            # 'use_norm': True,
            # system settings
            params["use_gpu"] = accelerator == True or accelerator == "gpu"
            m, last_metric = SOFTSPredictor.train(df, params, hpid, random_seed, True)
            m.cleanup()
        case _:
            if model.is_baseline(**params):
                tag = (
                    "baseline,univariate"
                    if covar_set_id == 0
                    else "baseline,multivariate"
                )
            params["h"] = args.future_steps
            params["max_steps"] = epochs
            params["validate"] = True
            params["random_seed"] = random_seed
            params["accelerator"] = (
                "auto" if accelerator == True or accelerator is None else accelerator
            )
            params["locks"] = locks
            last_metric = model.train(df, **params)

    fit_time = time.time() - start_time

    update_metrics_table(
        alchemyEngine,
        args.model,
        anchor_symbol,
        hpid,
        last_metric["epoch"] + 1,
        last_metric,
        fit_time,
        covar_set_id,
        topk_covar,
        tag,
        hps_id,
        ",".join([col for col in df.columns if col not in ("ds", "y")]),
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
    model,
    anchor_symbol,
    hpid,
    epochs,
    last_metric,
    fit_time,
    covar_set_id,
    topk_covar,
    tag,
    hps_id,
    covars,
):

    device_info = {}
    if "machine" in last_metric:
        device_info["machine"] = last_metric["machine"]
    if "device" in last_metric:
        device_info["device"] = last_metric["device"]
    device_info = json.dumps(device_info, sort_keys=True)

    def action():
        with alchemyEngine.begin() as conn:
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
                        sub_topk = :sub_topk,
                        device_info = :device_info,
                        covars = :covars
                    WHERE
                        model = :model
                        AND anchor_symbol = :anchor_symbol
                        AND hpid = :hpid
                        AND covar_set_id = :covar_set_id
                        AND hps_id = :hps_id
                """
                ),
                {
                    "model": model,
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
                    "hps_id": hps_id,
                    "device_info": device_info,
                    "covars": covars,
                },
            )

    for attempt in Retrying(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=5)
    ):
        with attempt:
            action()


def new_metric_keys(
    model, anchor_symbol, hpid, hyper_params, covar_set_id, hps_id, alchemyEngine
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
                        "model": model,
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
    alchemyEngine, model, symbol, hps_id, topk, ts_date, nan_limit
):
    # worker = get_worker()
    # alchemyEngine = worker.alchemyEngine
    match model:
        case "NeuralProphet":
            with_clause = """
                WITH univ_baseline_nc as (
                    select cov_table, cov_symbol, feature, symbol
                    from neuralprophet_corel
                    where 
                        symbol = %(symbol)s
                        and cov_symbol = symbol
                        and feature = 'y'
                        and ts_date = %(ts_date)s
                )
            """
            table = "neuralprophet_corel"
            model_cond = ""
        case _:
            with_clause = """
                WITH univ_baseline_nc as (
                    select cov_table, cov_symbol, feature, symbol
                    from paired_correlation
                    where 
                        model = %(model)s
                        and symbol = %(symbol)s
                        and cov_symbol = symbol
                        and feature = 'y'
                        and ts_date = %(ts_date)s
                )
            """
            table = "paired_correlation"
            model_cond = "and nc.model = %(model)s"
    query = f"""
        {with_clause},
        univ_baseline as (
            select
                hm.hyper_params, hm.mae, hm.rmse, hm.loss, hm.mae_val, 
                hm.rmse_val, hm.loss_val, hm.hpid, hm.epochs, hm.sub_topk,
                hm.covar_set_id, hm.anchor_symbol symbol,
                nc.cov_table, nc.cov_symbol, nc.feature
            from hps_metrics hm
            inner join univ_baseline_nc nc
                on hm.anchor_symbol = nc.symbol
            where 
                hm.model = %(model)s
                and hm.anchor_symbol = %(symbol)s 
                and hm.hps_id = %(hps_id)s
                and hm.covar_set_id = 0
        ),
        top_by_loss_val as (
            SELECT 
                ub.hyper_params, nc.mae, nc.rmse, nc.loss, nc.mae_val,
                nc.rmse_val, nc.loss_val, ub.hpid, nc.epochs, 1, 
                0, nc.symbol, nc.cov_table, nc.cov_symbol, nc.feature
            FROM {table} nc
            INNER JOIN
                univ_baseline ub
            ON nc.symbol = ub.symbol
            where 1=1
                {model_cond}
                and nc.ts_date = %(ts_date)s
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
            FROM {table} nc
            INNER JOIN
                univ_baseline ub
            ON nc.symbol = ub.symbol
            where 1=1
                {model_cond}
                and nc.ts_date = %(ts_date)s
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
        "model": model,
        "symbol": symbol,
        "hps_id": hps_id,
        "limit": topk,
        "ts_date": ts_date,
        "nan_limit": nan_limit,
    }

    df = pd.read_sql(query, alchemyEngine, params=params)
    df.drop("symbol", axis=1, inplace=True)

    return df


def get_topk_prediction_settings(alchemyEngine, model, symbol, hps_id, topk):
    # worker = get_worker()
    # alchemyEngine = worker.alchemyEngine

    query = """
        WITH baseline as (
            select loss_val
            from hps_metrics
            where 
                model = %(model)s
                and anchor_symbol = %(symbol)s 
                and hps_id = %(hps_id)s 
                and covar_set_id = 0
        ),
        top_by_loss_val AS (
            SELECT 
                hyper_params, mae, rmse, loss, mae_val, 
                rmse_val, loss_val, hpid, epochs, sub_topk,
                covar_set_id, covars
            FROM hps_metrics
            WHERE 
                model = %(model)s
                AND anchor_symbol = %(symbol)s
                AND hps_id = %(hps_id)s 
                and loss_val < (select loss_val from baseline)
            ORDER BY loss_val
            LIMIT %(limit)s
        ),
        top_by_loss AS (
            SELECT 
                hyper_params, mae, rmse, loss, mae_val, 
                rmse_val, loss_val, hpid, epochs, sub_topk,
                covar_set_id, covars
            FROM hps_metrics
            WHERE 
                model = %(model)s
                AND anchor_symbol = %(symbol)s
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
        "model": model,
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
        logger.warning(
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
    for holiday_name, dates in country_holidays.items():
        if date in dates:
            return holiday_name
    return None


def trim_forecasts_by_dates(forecast):
    future_date = forecast.iloc[-1]["ds"]
    days_count = (future_date - datetime.now()).days + 1
    start_date = future_date - timedelta(days=days_count * 2)
    forecast = forecast[forecast["ds"] >= start_date]

    return forecast


def distance(forecast):
    forecast_nona = forecast.dropna(subset=["yhat_n", "y"])
    mean_diff = (np.sqrt((forecast_nona["yhat_n"] - forecast_nona["y"]) ** 2)).mean()
    std_diff = (np.sqrt((forecast_nona["yhat_n"] - forecast_nona["y"]) ** 2)).std()
    return mean_diff, std_diff


def save_forecast_snapshot(
    alchemyEngine,
    model_name,
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
    mean_diff, std_diff = distance(forecast)
    forecast = trim_forecasts_by_dates(forecast)
    future_df = forecast[forecast["ds"] > datetime.now()][["ds", "yhat_n"]].copy()
    avg_yhat = future_df["yhat_n"].mean()
    future_df["plus_one"] = future_df["yhat_n"] / 100.0 + 1.0
    future_df["accumulated_returns"] = future_df["plus_one"].cumprod()
    cum_returns = (future_df["accumulated_returns"].iloc[-1] - 1.0) * 100.0

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
                        region,random_seed,future_steps,n_covars,cutoff_date,group_id,hpid,avg_loss,covar,
                        avg_yhat,cum_returns
                    )
                values(
                    :model,:symbol,:hyper_params,:covar_set_id,:mae_val,:rmse_val,:loss_val,
                    :mae,:rmse,:loss,:predict_diff_mean,:predict_diff_stddev,:epochs,:proc_time,
                    :mae_final,:rmse_final,:loss_final,:region,:random_seed,:future_steps,:n_covars,
                    :cutoff_date,:group_id,:hpid,:avg_loss,:covar,:avg_yhat,:cum_returns
                ) RETURNING id
                """
            ),
            {
                "model": model_name,
                "symbol": symbol,
                "hyper_params": hyper_params,
                "covar_set_id": covar_set_id,
                "mae_val": metrics["MAE_val"],
                "rmse_val": metrics["RMSE_val"],
                "loss_val": metrics["Loss_val"],
                "mae": metrics["MAE"],
                "rmse": metrics["RMSE"],
                "loss": metrics["Loss"],
                "predict_diff_mean": mean_diff,
                "predict_diff_stddev": std_diff,
                "epochs": metrics["epoch"] + 1,
                "proc_time": (
                    f"{str(proc_time)} seconds" if proc_time is not None else None
                ),
                "mae_final": metrics_final["MAE"],
                "rmse_final": metrics_final["RMSE"],
                "loss_final": metrics_final["Loss"],
                "region": region,
                "random_seed": random_seed,
                "future_steps": future_steps,
                "n_covars": n_covars,
                "cutoff_date": cutoff_date,
                "group_id": group_id,
                "hpid": hpid,
                "avg_loss": avg_loss,
                "covar": covar,
                "avg_yhat": avg_yhat,
                "cum_returns": cum_returns,
            },
        )
        snapshot_id = result.fetchone()[0]

        ## save to forecast_params table
        match model_name:
            case "NeuralProphet":
                forecast_params = forecast[
                    ["ds", "trend", "season_yearly", "yhat_n"]
                ].copy()
            case "SOFTS":
                forecast_params = forecast[["ds", "yhat_n"]].copy()
            case _:
                forecast_params = get_worker().model.trim_forecast(forecast)

        forecast_params.rename(columns={"ds": "date"}, inplace=True)
        forecast_params.loc[:, "symbol"] = symbol
        forecast_params.loc[:, "snapshot_id"] = snapshot_id
        # country_holidays = get_country_holidays(region)
        country_holidays = get_all_holidays(
            years=forecast_params["date"].dt.year.unique(), country=region
        )
        forecast_params.loc[:, "holiday"] = forecast_params["date"].apply(
            lambda x: check_holiday(x, country_holidays)
        )
        forecast_params.to_sql("forecast_params", conn, if_exists="append", index=False)

    return snapshot_id, len(forecast_params)


def train_predict(
    model,
    df,
    epochs,
    random_seed,
    early_stopping,
    country,
    validate,
    future_steps,
    **kwargs,
):
    try:
        m, metrics = train(
            model_name=model,
            df=df,
            country=country,
            epochs=epochs,
            random_seed=random_seed,
            early_stopping=early_stopping,
            validate=validate,
            n_forecasts=future_steps,
            save_model_file=True,
            **kwargs,
        )

        match model:
            case "NeuralProphet":
                forecast = NPPredictor.predict(m, df, random_seed, future_steps)
            case "SOFTS":
                forecast = SOFTSPredictor.predict(m, df, country)
                m.cleanup()
            case _:
                forecast = get_worker().model.predict(
                    df, random_seed=random_seed, h=future_steps
                )
    except Exception as e:
        get_logger().error(traceback.format_exc())
        raise e

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


def forecast(
    model, symbol, df, ranked_features, hps_metric, region, cutoff_date, group_id
):
    worker = get_worker()
    alchemyEngine, logger, args = worker.alchemyEngine, worker.logger, worker.args

    hyperparams = hps_metric["hyper_params"]
    covar_set_id = int(hps_metric["covar_set_id"])
    hp_str = json.dumps(hyperparams, sort_keys=True)
    logger.info(
        (
            "%s - forecasting with setting:\n"
            "symbol: %s\n"
            "hpid: %s\n"
            "loss: %s\n"
            "loss_val: %s\n"
            "covar_set_id: %s\n"
            "sub_topk: %s\n"
            "cutoff_date: %s\n"
            "hyper-params: %s"
        ),
        model,
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
            df,
            hps_metric["cov_table"],
            hps_metric["cov_symbol"],
            hps_metric["feature"],
            df["ds"].min().strftime("%Y-%m-%d"),
            alchemyEngine,
        )
    else:
        if hps_metric["sub_topk"] is None:
            new_df = df
        else:
            cols = ["ds", "y"] + [
                covar.strip() for covar in hps_metric["covars"].split(",")
            ]
            new_df = df[cols]
            # new_df = select_topk_features(
            #     df, ranked_features, int(hps_metric["sub_topk"])
            # )

    logger.info(
        "dataframe augmented with covar_set_id %s: %s", covar_set_id, new_df.shape
    )

    if "topk_covar" in hyperparams:
        hyperparams.pop("topk_covar")
    if "random_seed" not in hyperparams:
        hyperparams["random_seed"] = args.random_seed

    start_time = time.time()
    with worker_client() as client:
        futures = []
        # train with validation
        futures.append(
            client.submit(
                train,
                model_name=model,
                symbol=symbol,
                df=new_df,
                epochs=args.epochs,
                early_stopping=args.early_stopping,
                weekly_seasonality=False,
                daily_seasonality=False,
                impute_missing=True,
                accelerator="gpu" if args.accelerator else None,
                validate=True,
                country=region,
                changepoints_range=1.0,
                # save_model_file=True,
                **hyperparams,
            )
        )
        # train with full dataset without validation split
        # worker_mem_needed = measure_needed_mem(new_df, hyperparams)
        futures.append(
            client.submit(
                train_predict,
                model=model,
                df=new_df,
                country=region,
                epochs=args.epochs,
                # random_seed=args.random_seed,
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

    avg_loss = round((metrics["Loss_val"] + metrics_final["Loss"]) / 2.0, 5)

    covar_columns = [col for col in new_df.columns if col not in ("ds", "y")]
    n_covars = len(covar_columns)

    snapshot_id, horizons = save_forecast_snapshot(
        alchemyEngine,
        model,
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
        ",".join(covar_columns),
    )

    logger.info(
        "%s - estimated %s horizons. snapshot_id: %s, averaged loss: %s",
        symbol,
        horizons,
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
    random_seed,
    future_steps,
    topk,
    hps_id,
    cutoff_date,
    ranked_features,
    df,
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
    s1 = get_topk_prediction_settings(alchemyEngine, args.model, symbol, hps_id, topk)
    # get univariate and 2*topk 2-pair covariate settings
    nan_threshold = round(len(df) * args.nan_limit, 0)
    s2 = get_topk_foundation_settings(
        alchemyEngine, args.model, symbol, hps_id, topk, cutoff_date, nan_threshold
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
                args.model,
                symbol,
                df,
                ranked_features,
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
        args.model,
        symbol,
        forecasts,
        avg_loss,
        weights,
        region,
        snapshot_ids,
        random_seed,
        future_steps,
        group_id,
        cutoff_date,
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
            merged_df, params, covar_set_id = get_best_prediction_setting(
                alchemyEngine, logger, symbol, df, topk, i
            )
            dfs.append(merged_df)
            params_json = json.dumps(params.copy(), sort_keys=True)
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
    model,
    symbol,
    top_forecasts,
    avg_loss,
    weights,
    region,
    snapshot_ids,
    random_seed,
    future_steps,
    group_id,
    cutoff_date,
):
    ens_df = None
    # extract unique years from top_forecasts
    years = list(
        set(
            [
                year
                for df in top_forecasts
                for year in df["ds"].dt.year.unique().tolist()
            ]
        )
    )
    country_holidays = get_all_holidays(years=years, country=region)
    # country_holidays = get_country_holidays(region)
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
    avg_yhat = ens_df["yhat_n"].mean()
    ens_df["plus_one"] = ens_df["yhat_n"] / 100.0 + 1.0
    ens_df["accumulated_returns"] = ens_df["plus_one"].cumprod()
    cum_returns = (ens_df["accumulated_returns"].iloc[-1] - 1.0) * 100.0
    ens_df.drop(columns=["plus_one", "accumulated_returns"], inplace=True)

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
                        group_id,cutoff_date,avg_yhat,cum_returns
                    )
                values(
                    :model,:symbol,:hyper_params,
                    :region,:random_seed,:future_steps,
                    :group_id,:cutoff_date,:avg_yhat,:cum_returns
                ) RETURNING id
                """
            ),
            {
                "model": f"{model}_Ensemble",
                "symbol": symbol,
                "hyper_params": hyper_params,
                "region": region,
                "random_seed": random_seed,
                "future_steps": future_steps,
                "group_id": group_id,
                "cutoff_date": cutoff_date,
                "avg_yhat": avg_yhat,
                "cum_returns": cum_returns,
            },
        )
        snapshot_id = result.fetchone()[0]
        ens_df.loc[:, "snapshot_id"] = snapshot_id
        ens_df.to_sql("forecast_params", conn, if_exists="append", index=False)


def init_hps(hps, model, symbol, args, client, alchemyEngine, logger):
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
    hps.model = model

    return args


def count_topk_hp(alchemyEngine, model, hps_id, base_loss):
    with alchemyEngine.connect() as conn:
        result = conn.execute(
            text(
                """
                    select count(*)
                    from hps_metrics
                    where 
                        model = :model
                        and hps_id = :hps_id
                        and loss_val <= :base_loss
                """
            ),
            {
                "model": model,
                "hps_id": hps_id,
                "base_loss": base_loss,
            },
        )
        return result.fetchone()[0]


def _search_space(model_name, model, max_covars):
    match model_name:
        case "NeuralProphet":
            ss = f"""dict(
                growth=["linear", "discontinuous"],
                batch_size=[None, 100, 200, 300, 400, 500],
                n_lags=range(0, 60+1),
                yearly_seasonality=["auto"] + list(range(1, 60+1)),
                ar_layer_spec=[None] + [[2**w, d] for w in range(1, 10+1) for d in range(1, 64+1)],
                ar_reg=uniform(0, 100),
                lagged_reg_layer_spec=[None] + [[2**w, d] for w in range(1, 10+1) for d in range(1, 64+1)],
                topk_covar=range(0, {max_covars}+1),
                optimizer=["AdamW", "SGD"],
                trend_reg=uniform(0, 100),
                trend_reg_threshold=[True, False],
                seasonality_reg=uniform(0, 100),
                seasonality_mode=["additive", "multiplicative"],
                normalize=["off", "standardize", "soft", "soft1"],
            )"""
        case "SOFTS":
            # d_model=[2**w for w in range(5, 8+1)],
            # d_ff=[2**w for w in range(5, 8+1)],
            # d_model_d_ff=[2**w for w in range(5, 8+1)],
            ss = f"""dict(
                seq_len=range(5, 300+1),
                d_model=[2**w for w in range(6, 9+1)],
                d_core=[2**w for w in range(5, 10+1)],
                d_ff=[2**w for w in range(6, 10+1)],
                e_layers=range(4, 16+1),
                learning_rate=loguniform(0.0001, 0.002),
                lradj=["type1", "type2", "constant", "cosine"],
                patience=range(3, 10+1),
                batch_size=[2**w for w in range(5, 8+1)],
                dropout=uniform(0, 0.5),
                activation=["relu","gelu","relu6","elu","selu","celu","leaky_relu","prelu","rrelu","glu"],
                use_norm=[True, False],
                optimizer=["Adam", "AdamW", "SGD"],
                topk_covar=list(range(0, {max_covars}+1)),
                covar_dist=dirichlet([1.0]*{max_covars}),
            )"""
        case _:
            return model.search_space(max_covars=max_covars)
    return ss


def fast_bayesopt(
    model,
    client,
    alchemyEngine,
    logger,
    df,
    covar_set_id,
    hps_id,
    ranked_features,
    base_loss,
    args,
):
    # worker = get_worker()
    # logger = worker.logger

    from scipy.stats import uniform, loguniform, dirichlet
    from marten.models.hp_search import (
        _cleanup_stale_keys,
        _bayesopt_run,
        update_hps_sessions,
    )

    _cleanup_stale_keys()

    space_str = _search_space(
        args.model, model, min(args.max_covars, len(ranked_features))
    )

    # Convert args to a dictionary, excluding non-serializable items
    args_dict = {k: v for k, v in vars(args).items() if not callable(v)}
    args_json = json.dumps(args_dict, sort_keys=True)
    update_hps_sessions(hps_id, "fast_bayesopt", args_json, space_str, covar_set_id)

    n_jobs = args.batch_size

    domain_size = args.domain_size
    base_ds = n_jobs * args.mini_itr * args.max_itr
    if domain_size < 0:
        domain_size = None
    elif domain_size == 0:
        domain_size = base_ds * 10
    elif domain_size < base_ds:
        domain_size = base_ds

    if args.model == "SOFTS" or (model is not None and not model.accept_missing_data()):
        df, _ = impute(df, args.random_seed, client)

    locks = get_accelerator_locks(cpu_leases=2, gpu_leases=1, timeout="10s")
    # split large iterations into smaller runs to avoid OOM / memory leak
    for i in range(args.max_itr):
        logger.info(
            "running bayesopt mini-iteration %s/%s batch size: %s runs: %s",
            i + 1,
            args.max_itr,
            n_jobs,
            args.mini_itr,
        )
        min_loss = _bayesopt_run(
            df,
            n_jobs,
            covar_set_id,
            hps_id,
            ranked_features,
            eval(
                space_str,
                {"uniform": uniform, "loguniform": loguniform, "dirichlet": dirichlet},
            ),
            args,
            args.mini_itr,
            domain_size,
            args.resume or i > 0,
            locks,
        )

        if min_loss is None or min_loss > base_loss:
            continue
        topk_count = count_topk_hp(alchemyEngine, args.model, hps_id, base_loss)

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
            client.restart()


def update_covar_set_id(alchemyEngine, hps_id, covar_set_id):
    sql = """
        update hps_sessions
        set covar_set_id = :covar_set_id
        where id = :hps_id
    """
    with alchemyEngine.begin() as conn:
        conn.execute(text(sql), {"hps_id": hps_id, "covar_set_id": covar_set_id})


def _univariate_default_hp(model, client, anchor_df, args, hps_id):
    df = anchor_df[["ds", "y"]]
    params = None
    match args.model:
        case "NeuralProphet":
            from marten.models.hp_search import default_params

            params = default_params
            # default_params = default_params
        case "SOFTS":
            params = baseline_config
        case _:
            params = model.baseline_params()
    return client.submit(
        log_metrics_for_hyper_params,
        args.symbol,
        df,
        params,
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
    ).result()


def min_covar_loss_val(alchemyEngine, model, symbol, ts_date):
    with alchemyEngine.connect() as conn:
        match model:
            case "NeuralProphet":
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
            case _:
                result = conn.execute(
                    text(
                        """
                            select min(loss_val)
                            from paired_correlation
                            where 
                                model = :model
                                and symbol = :symbol 
                                and ts_date = :ts_date
                        """
                    ),
                    {
                        "model": model,
                        "symbol": symbol,
                        "ts_date": ts_date,
                    },
                )
        return result.fetchone()[0]


def covars_and_search(model, client, symbol, alchemyEngine, logger, args):
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

    args = init_hps(hps, model, symbol, args, client, alchemyEngine, logger)
    cutoff_date = _get_cutoff_date(args)
    anchor_df, anchor_table = load_anchor_ts(
        args.symbol, args.timestep_limit, alchemyEngine, cutoff_date
    )
    cutoff_date = anchor_df["ds"].max().strftime("%Y-%m-%d")

    hps_id, covar_set_id = get_hps_session(
        args.symbol, args.model, cutoff_date, args.resume, len(anchor_df)
    )
    args.covar_set_id = covar_set_id
    logger.info(
        "HPS session ID: %s, Model: %s, Cutoff date: %s, CovarSet ID: %s, Anchor Table: %s",
        hps_id,
        args.model,
        cutoff_date,
        covar_set_id,
        anchor_table,
    )

    univ_loss = _univariate_default_hp(model, client, anchor_df, args, hps_id)

    min_covar_loss = min_covar_loss_val(alchemyEngine, args.model, symbol, cutoff_date)
    min_covar_loss = min_covar_loss if min_covar_loss is not None else LOSS_CAP

    base_loss = min(float(univ_loss) * args.loss_quantile, min_covar_loss)

    # if in resume mode, check if the topk HP is present, and further check if prediction is already conducted.
    topk_count = count_topk_hp(alchemyEngine, args.model, hps_id, base_loss)
    if args.resume and topk_count >= args.topk:
        logger.info(
            "Found %s HP with Loss_val less than %s in HP search history already. Skipping covariate and HP search.",
            topk_count,
            base_loss,
        )
        df, covar_set_id, ranked_features = augment_anchor_df_with_covars(
            anchor_df, args, alchemyEngine, logger, cutoff_date
        )
        # df_future = client.scatter(df)
        # ranked_features_future = client.scatter(ranked_features)
        return hps_id, cutoff_date, ranked_features, df
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
    # logger.info("waiting dask futures: %s", len(hps.futures))
    await_futures(hps.futures, hard_wait=True)

    logger.info(
        "%s covariate baseline metric computation completed. Time taken: %s seconds",
        args.symbol,
        round(time.time() - t1_start, 3),
    )

    min_covar_loss = min_covar_loss_val(alchemyEngine, args.model, symbol, cutoff_date)
    min_covar_loss = min_covar_loss if min_covar_loss is not None else LOSS_CAP
    base_loss = min(base_loss, min_covar_loss)

    # run HP search using Bayeopt and check whether needed HP(s) are found
    logger.info(
        "Starting Bayesian optimization search for hyper-parameters. Loss_val threshold: %s",
        round(base_loss, 5),
    )

    # scale-in to preserve more memory for hps
    worker_size = (
        args.min_worker
        if args.model != "NeuralProphet"
        else max(args.min_worker, round(args.max_worker * 0.5))
    )
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
        model,
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

    return hps_id, cutoff_date, ranked_features, df
