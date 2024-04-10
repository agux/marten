import time
import pandas as pd
import json
import hashlib
import warnings

from sqlalchemy import text

from neuralprophet import NeuralProphet, set_random_seed, set_log_level

from tenacity import (
    stop_after_attempt,
    wait_exponential,
    Retrying,
)

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
):
    # Local import of get_worker to avoid circular import issue
    from dask.distributed import get_worker
    worker = get_worker()
    alchemyEngine, logger = worker.alchemyEngine, worker.logger
    if anchor_symbol == cov_symbol:
        if feature == "y":
            # no covariate is needed. this is a baseline metric
            merged_df = anchor_df[["ds", "y"]]
        else:
            # using endogenous features as covariate
            merged_df = anchor_df[["ds", "y", feature]]
    else:
        # `cov_symbol` may contain special characters such as `.IXIC`, or `H-FIN`. The dot and hyphen is not allowed in column alias.
        # Convert common special characters often seen in stock / index symbols to valid replacements as PostgreSQL table column alias.
        cov_symbol_sanitized = cov_symbol.replace(".", "_").replace("-", "_")
        cov_symbol_sanitized = f"{feature}_{cov_symbol_sanitized}"
        if cov_table != "bond_metrics_em":
            query = f"""
                    select date ds, {feature} {cov_symbol_sanitized}
                    from {cov_table}
                    where symbol = %(cov_symbol)s
                    and date >= %(min_date)s
                    order by date
                """
            params = {
                "cov_symbol": cov_symbol,
                "min_date": min_date,
            }
        else:
            query = f"""
                    select date ds, {feature} {cov_symbol_sanitized}
                    from {cov_table}
                    where date >= %(min_date)s
                    order by date
                """
            params = {
                "min_date": min_date,
            }
        cov_symbol_df = pd.read_sql(
            query, alchemyEngine, params=params, parse_dates=["ds"]
        )
        if cov_symbol_df.empty:
            return None
        merged_df = pd.merge(anchor_df, cov_symbol_df, on="ds", how="left")

    start_time = time.time()
    metrics = None
    try:
        metrics = train(
            df=merged_df,
            epochs=None,
            random_seed=random_seed,
            early_stopping=early_stopping,
            batch_size=None,
            weekly_seasonality=False,
            daily_seasonality=False,
            impute_missing=True,
            accelerator=accelerator,
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
    nan_count = merged_df.iloc[:, -1].isna().sum()
    # Assuming `nan_count` is a numpy.int64 value
    nan_count = int(nan_count)  # Convert to Python's native int type
    save_covar_metrics(
        anchor_symbol,
        cov_table,
        cov_symbol,
        feature,
        last_row,
        fit_time,
        timesteps,
        nan_count,
        alchemyEngine,
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
    alchemyEngine,
):
    # Insert data into the table
    with alchemyEngine.begin() as conn:
        # Inserting DataFrame into the database table
        for index, row in cov_metrics.iterrows():
            conn.execute(
                text(
                    """
                    INSERT INTO neuralprophet_corel 
                    (symbol, cov_table, cov_symbol, feature, mae_val, rmse_val, loss_val, fit_time, timesteps, nan_count) 
                    VALUES (:symbol, :cov_table, :cov_symbol, :feature, :mae_val, :rmse_val, :loss_val, :fit_time, :timesteps, :nan_count) 
                    ON CONFLICT (symbol, cov_symbol, feature, cov_table) 
                    DO UPDATE SET 
                        mae_val = EXCLUDED.mae_val, 
                        rmse_val = EXCLUDED.rmse_val, 
                        loss_val = EXCLUDED.loss_val,
                        fit_time = EXCLUDED.fit_time,
                        timesteps = EXCLUDED.timesteps,
                        nan_count = EXCLUDED.nan_count
                """
                ),
                {
                    "symbol": anchor_symbol,
                    "cov_table": cov_table,
                    "cov_symbol": cov_symbol,
                    "feature": feature,
                    "mae_val": row["MAE_val"],
                    "rmse_val": row["RMSE_val"],
                    "loss_val": row["Loss_val"],
                    "fit_time": (str(fit_time) + " seconds"),
                    "timesteps": timesteps,
                    "nan_count": nan_count,
                },
            )


def train(df, epochs=None, random_seed=7, early_stopping=True, **kwargs):
    set_log_level("ERROR")
    set_random_seed(random_seed)

    with warnings.catch_warnings():
        # suppress swarming warning:
        # WARNING - (py.warnings._showwarnmsg) - 
        # ....../.pyenv/versions/3.12.2/envs/venv_3.12.2/lib/python3.12/site-packages/neuralprophet/df_utils.py:1152: 
        # FutureWarning: Series.view is deprecated and will be removed in a future version. Use ``astype`` as an alternative to change the dtype.
        # converted_ds = pd.to_datetime(ds_col, utc=True).view(dtype=np.int64)
        warnings.simplefilter("ignore", FutureWarning)

        m = NeuralProphet(**kwargs)
        covars = [col for col in df.columns if col not in ("ds", "y")]
        m.add_lagged_regressor(covars)
        train_df, test_df = m.split_df(
            df,
            valid_p=1.0 / 10,
            freq="D",
        )
        try:
            metrics = m.fit(
                train_df,
                validation_df=test_df,
                progress=None,
                epochs=epochs,
                early_stopping=early_stopping,
                freq="D",
            )
            return metrics
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


def log_metrics_for_hyper_params(
    anchor_symbol,
    df,
    params,
    epochs,
    random_seed,
    accelerator,
    covar_set_id,
    early_stopping,
):
    # Local import of get_worker to avoid circular import issue
    from dask.distributed import get_worker

    worker = get_worker()
    alchemyEngine, logger = worker.alchemyEngine, worker.logger

    # to support distributed processing, we try to insert a new record (with primary keys only)
    # into grid_search_metrics first. If we hit duplicated key error, return None.
    # Otherwise we could proceed further code execution.
    param_str = json.dumps(params)
    hpid = hashlib.md5(param_str.encode("utf-8")).hexdigest()
    if not new_metric_keys(
        anchor_symbol, hpid, param_str, covar_set_id, alchemyEngine
    ):
        logger.debug("Skip re-entry for %s: %s", anchor_symbol, param_str)
        return None

    start_time = time.time()
    metrics = None
    try:
        metrics = train(
            df,
            epochs=epochs,
            random_seed=random_seed,
            early_stopping=early_stopping,
            batch_size=params["batch_size"],
            n_lags=params["n_lags"],
            yearly_seasonality=params["yearly_seasonality"],
            ar_layers=params["ar_layers"],
            lagged_reg_layers=params["lagged_reg_layers"],
            weekly_seasonality=False,
            daily_seasonality=False,
            impute_missing=True,
            accelerator=accelerator,
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
    covars = [col for col in df.columns if col not in ("ds", "y")]
    logger.info("params:%s\n#covars:%s\n%s", params, len(covars), last_metric)

    update_metrics_table(
        alchemyEngine,
        params,
        anchor_symbol,
        hpid,
        last_metric["epoch"] + 1,
        last_metric,
        fit_time,
        covar_set_id,
    )

    return last_metric


def update_metrics_table(
    alchemyEngine,
    params,
    anchor_symbol,
    hpid,
    epochs,
    last_metric,
    fit_time,
    covar_set_id,
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
                    UPDATE grid_search_metrics
                    SET 
                        mae_val = :mae_val, 
                        rmse_val = :rmse_val, 
                        loss_val = :loss_val, 
                        mae = :mae,
                        rmse = :rmse,
                        loss = :loss,
                        fit_time = :fit_time,
                        epochs = :epochs,
                        tag = :tag
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
                    "fit_time": (str(fit_time) + " seconds"),
                    "epochs": epochs,
                },
            )

    for attempt in Retrying(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=5)
    ):
        with attempt:
            action()


def new_metric_keys(anchor_symbol, hpid, hyper_params, covar_set_id, alchemyEngine):
    def action():
        try:
            with alchemyEngine.begin() as conn:
                conn.execute(
                    text(
                        """
                        INSERT INTO grid_search_metrics (model, anchor_symbol, hpid, hyper_params, covar_set_id) 
                        VALUES (:model, :anchor_symbol, :hpid, :hyper_params, :covar_set_id)
                        """
                    ),
                    {
                        "model": "NeuralProphet",
                        "anchor_symbol": anchor_symbol,
                        "hpid": hpid,
                        "hyper_params": hyper_params,
                        "covar_set_id": covar_set_id,
                    },
                )
                return True
        except Exception as e:
            if "duplicate key value violates unique constraint" in str(e):
                return False
            else:
                raise

    for attempt in Retrying(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=5)
    ):
        with attempt:
            return action()
