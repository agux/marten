import time
import pandas as pd
import json
import hashlib
import warnings
import math
import torch

from datetime import datetime, timedelta

from sqlalchemy import text

from neuralprophet import NeuralProphet, set_random_seed, set_log_level
from neuralprophet.hdays_utils import get_country_holidays

from dask.distributed import get_worker, worker_client

from tenacity import (
    stop_after_attempt,
    wait_exponential,
    Retrying,
    retry_if_exception_type,
    RetryError,
)

from marten.utils.holidays import get_holiday_region
from marten.utils.logger import get_logger
from marten.utils.pl import GlobalProgressBar

nan_inf_replacement = 99999.99999

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
    
    # `cov_symbol` may contain special characters such as `.IXIC`, or `H-FIN`. The dot and hyphen is not allowed in column alias.
    # Convert common special characters often seen in stock / index symbols to valid replacements as PostgreSQL table column alias.
    # cov_symbol_sanitized = cov_symbol.replace(".", "_").replace("-", "_")
    # cov_symbol_sanitized = f"{feature}_{cov_symbol_sanitized}"
    cov_symbol_sanitized = f"{feature}_{cov_symbol}"

    match cov_table:
        case "bond_metrics_em" | "bond_metrics_em_view" | "currency_boc_safe_view":
            query = f"""
                select date ds, {feature} "{cov_symbol_sanitized}"
                from {cov_table}
                where date >= %(min_date)s
                order by date
            """
            params = {
                "min_date": min_date,
            }
        case _:
            query = f"""
                select date ds, {feature} "{cov_symbol_sanitized}"
                from {cov_table}
                where symbol = %(cov_symbol)s
                and date >= %(min_date)s
                order by date
            """
            params = {
                "cov_symbol": cov_symbol,
                "min_date": min_date,
            }

    with alchemyEngine.connect() as conn:
        cov_symbol_df = pd.read_sql(
            query, conn, params=params, parse_dates=["ds"]
        )

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
    alchemyEngine, logger = worker.alchemyEngine, worker.logger
    merged_df = merge_covar_df(
        anchor_symbol,
        anchor_df,
        cov_table,
        cov_symbol,
        feature,
        min_date,
        alchemyEngine,
    )
    if merged_df is None:
        # FIXME: sometimes merged_df is None even if there's data in table
        logger.info("skipping covariate: %s, %s, %s, %s", cov_table, cov_symbol, feature, min_date)
        return None

    start_time = time.time()
    metrics = None
    region = None
    if infer_holiday:
        region = get_holiday_region(alchemyEngine, anchor_symbol)
    try:
        _, metrics = train(
            df=merged_df,
            epochs=None,
            random_seed=random_seed,
            early_stopping=early_stopping,
            batch_size=None,
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
        for _, row in cov_metrics.iterrows():
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

def log_retry(retry_state):
    if retry_state.outcome.failed:
        exception = retry_state.outcome.exception()
        get_logger().warning(
            f"Retrying, attempt {retry_state.attempt_number} after exception: {exception}"
        )

def _trainer_config():
    #TODO: can we use custom trainer config to specify gpu and device?
    config = {
        "callbacks": [
            GlobalProgressBar(False, False),    # suppress/disable progress bar display
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

    m = NeuralProphet(trainer_config=_trainer_config(), **kwargs)
    # m = NeuralProphet(**kwargs)
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
                # progress=None,
                epochs=epochs,
                early_stopping=early_stopping,
                freq="B",
            )
        else:
            metrics = m.fit(
                df,
                # progress=None,
                epochs=epochs,
                early_stopping=early_stopping,
                freq="B",
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

def train(
    df,
    epochs=None,
    random_seed=7,
    early_stopping=True,
    country=None,
    validate=True,
    **kwargs,
):
    def _train_with_cpu():
        if kwargs.get("accelerator") in ("cpu", "auto"):
            kwargs.pop('accelerator')
            m, metrics = _try_fitting(df,
                    epochs,
                    random_seed,
                    early_stopping,
                    country,
                    validate,
                    **kwargs)
            return m, metrics
        
    with warnings.catch_warnings():
        # suppress swarming warning:
        # WARNING - (py.warnings._showwarnmsg) -
        # ....../.pyenv/versions/3.12.2/envs/venv_3.12.2/lib/python3.12/site-packages/neuralprophet/df_utils.py:1152:
        # FutureWarning: Series.view is deprecated and will be removed in a future version. Use ``astype`` as an alternative to change the dtype.
        # converted_ds = pd.to_datetime(ds_col, utc=True).view(dtype=np.int64)
        warnings.simplefilter("ignore", FutureWarning)

        try:
            for attempt in Retrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, max=5),
                retry=retry_if_exception_type(torch.cuda.OutOfMemoryError),
                before_sleep=log_retry,
            ):
                with attempt:
                    m, metrics = _try_fitting(df,
                        epochs,
                        random_seed,
                        early_stopping,
                        country,
                        validate,
                        **kwargs)
                    return m, metrics
        except RetryError as e:
            if "OutOfMemoryError" in str(e):
                # final attempt to train on CPU
                # remove `accelerator` parameter from **kwargs
                get_logger().warning(
                    "falling back to CPU due to torch.cuda.OutOfMemoryError"
                )
                return _train_with_cpu()
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
    infer_holiday,
):
    worker = get_worker()
    alchemyEngine, logger = worker.alchemyEngine, worker.logger

    # to support distributed processing, we try to insert a new record (with primary keys only)
    # into hps_metrics first. If we hit duplicated key error, return None.
    # Otherwise we could proceed further code execution.
    param_str = json.dumps(params, sort_keys=True)
    hpid = hashlib.md5(param_str.encode("utf-8")).hexdigest()
    if not new_metric_keys(anchor_symbol, hpid, param_str, covar_set_id, alchemyEngine):
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
                        and covar_set_id = :covar_set_id
                    """
                ),
                {
                    "model": "NeuralProphet",
                    "anchor_symbol": anchor_symbol,
                    "hpid": hpid,
                    # "hyper_params": param_str,
                    "covar_set_id": covar_set_id,
                },
            )
            row = result.fetchone()
            if row is not None:
                loss_val = row[0]
            return loss_val

    start_time = time.time()
    metrics = None
    region = None
    if infer_holiday:
        region = get_holiday_region(alchemyEngine, anchor_symbol)
    try:
        _, metrics = train(
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
            validate=True,
            country=region,
            changepoints_range=1.0,
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
    last_metric["Loss_val"] = sanitize_loss(last_metric["Loss_val"])
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
    )

    return last_metric["Loss_val"]


def sanitize_loss(value):
    global nan_inf_replacement
    return nan_inf_replacement if math.isnan(value) or math.isinf(value) else value


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
                        INSERT INTO hps_metrics (model, anchor_symbol, hpid, hyper_params, covar_set_id) 
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


def get_best_prediction_setting(alchemyEngine, logger, symbol, timestep_limit):
    # find the model setting with optimum performance, including univariate default setting.
    from marten.models.hp_search import (
        default_params,
        load_anchor_ts,
        augment_anchor_df_with_covars,
    )
    from types import SimpleNamespace

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
                    hyper_params,
                    loss_val,
                    covar_set_id,
                    null nan_count
                from
                    hps_metrics
                where
                    anchor_symbol = %(symbol)s
                order by
                    loss_val asc
                limit 1
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
                    nan_count
                from
                    neuralprophet_corel nc
                where
                    symbol = %(symbol)s
                order by
                    loss_val asc
                limit 1
            )
        )
        order by
            loss_val asc
        limit 1
    """
    params = {
        "symbol": symbol,
    }

    with alchemyEngine.connect() as conn:
        best_setting = pd.read_sql(query, conn, params=params)

    df, _ = load_anchor_ts(symbol, timestep_limit, alchemyEngine)
    hyperparams = None
    covar_set_id = None
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
        df = df[["ds", "y"]]
    elif best_setting["hyper_params"].iloc[0] is not None:
        best_setting = best_setting.iloc[0]
        covar_set_id = int(best_setting["covar_set_id"])
        logger.info(
            (
                "%s - found best setting in hps_metrics:\n"
                "loss_val: %s\n"
                "covar_set_id: %s"
            ),
            symbol,
            best_setting["loss_val"],
            covar_set_id,
        )

        hyperparams = json.loads(best_setting["hyper_params"])
        df, _ = augment_anchor_df_with_covars(
            df,
            SimpleNamespace(covar_set_id=covar_set_id, symbol=symbol),
            alchemyEngine,
            logger,
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
                "nan_count: %s"
            ),
            symbol,
            best_setting["loss_val"],
            best_setting["cov_table"],
            best_setting["cov_symbol"],
            best_setting["feature"],
            best_setting["nan_count"],
        )
        hyperparams = default_params
        df = merge_covar_df(
            symbol,
            df[["ds", "y"]],
            best_setting["cov_table"],
            best_setting["cov_symbol"],
            best_setting["feature"],
            df["ds"].min().strftime("%Y-%m-%d"),
            alchemyEngine,
        )
    return df, hyperparams, covar_set_id


# Function to check if a date is a holiday and return the holiday name
def check_holiday(date, country_holidays):
    return country_holidays.get(date) if date in country_holidays else None


def save_forecast_snapshot(
    alchemyEngine,
    symbol,
    hyper_params,
    covar_set_id,
    metrics,
    metrics_final,
    forecast,
    fit_time,
    region,
    random_seed,
    future_steps,
    n_covars,
):
    metric = metrics.iloc[-1]
    metric_final = metrics_final.iloc[-1]
    mean_diff = (forecast["yhat1"] - forecast["y"]).mean()
    std_diff = (forecast["yhat1"] - forecast["y"]).std()

    with alchemyEngine.begin() as conn:
        result = conn.execute(
            text(
                """
                insert
                    into
                    predict_snapshots
                    (
                        model,symbol,hyper_params,covar_set_id,mae_val,rmse_val,loss_val,mae,rmse,loss,
                        predict_diff_mean,predict_diff_stddev,epochs,fit_time,mae_final,rmse_final,loss_final,
                        region,random_seed,future_steps,n_covars
                    )
                values(
                    :model,:symbol,:hyper_params,:covar_set_id,:mae_val,:rmse_val,:loss_val,
                    :mae,:rmse,:loss,:predict_diff_mean,:predict_diff_stddev,:epochs,:fit_time,
                    :mae_final,:rmse_final,:loss_final,:region,:random_seed,:future_steps,:n_covars
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
                "fit_time": f"{str(fit_time)} seconds",
                "mae_final": metric_final["MAE"],
                "rmse_final": metric_final["RMSE"],
                "loss_final": metric_final["Loss"],
                "region": region,
                "random_seed": random_seed,
                "future_steps": future_steps,
                "n_covars": n_covars,
            },
        )
        snapshot_id = result.fetchone()[0]

        ## save to ft_yearly_params table
        future_date = forecast.iloc[-1]["ds"]
        days_count = (future_date - datetime.now()).days + 1
        start_date = future_date - timedelta(days=days_count * 2)
        forecast = forecast[forecast["ds"] >= start_date]

        country_holidays = get_country_holidays(region)

        forecast_params = forecast[["ds", "trend", "season_yearly"]]
        forecast_params.rename(columns={"ds": "date"}, inplace=True)
        forecast_params.loc[:, "symbol"] = symbol
        forecast_params.loc[:, "snapshot_id"] = snapshot_id
        forecast_params.loc[:, "holiday"] = forecast_params["date"].apply(
            lambda x: check_holiday(x, country_holidays)
        )
        forecast_params.to_sql("forecast_params", conn, if_exists="append", index=False)

    return snapshot_id, len(forecast_params)


def predict_best(
    symbol,
    early_stopping,
    timestep_limit,
    epochs,
    random_seed,
    future_steps,
):
    worker = get_worker()
    alchemyEngine, logger = worker.alchemyEngine, worker.logger

    df, params, covar_set_id = get_best_prediction_setting(
        alchemyEngine, logger, symbol, timestep_limit
    )
    logger.info(
        "%s - using hyper-parameters:\n%s", symbol, json.dumps(params, sort_keys=True)
    )
    region = get_holiday_region(alchemyEngine, symbol)
    logger.info("%s - inferred holiday region: %s", symbol, region)

    # use the best performing setting to fit and then predict
    start_time = time.time()
    futures = []
    with worker_client() as client:
        # train with validation
        futures.append(client.submit(
                train,
                df=df,
                country=region,
                epochs=epochs,
                random_seed=random_seed,
                early_stopping=early_stopping,
                weekly_seasonality=False,
                daily_seasonality=False,
                impute_missing=True,
                validate=True,
                changepoints_range=1.0,
                **params,
            )
        )
        # train with full dataset without validation split
        futures.append(client.submit(
            train,
            df=df,
            country=region,
            epochs=epochs,
            random_seed=random_seed,
            early_stopping=early_stopping,
            weekly_seasonality=False,
            daily_seasonality=False,
            impute_missing=True,
            validate=False,
            n_forecasts=future_steps,
            changepoints_range=1.0,
            **params,
        ))
        results = client.gather(futures)

    metrics = results[0][1]
    m, metrics_final = results[1][0], results[1][1]

    fit_time = time.time() - start_time

    set_log_level("ERROR")
    set_random_seed(random_seed)

    future = m.make_future_dataframe(
        df, n_historic_predictions=True, periods=future_steps
    )
    forecast = m.predict(future)

    n_covars = len([col for col in df.columns if col not in ("ds", "y")])

    # save the snapshot and yearly seasonality coefficients to tables.
    snapshot_id, n_yearly_seasonality = save_forecast_snapshot(
        alchemyEngine,
        symbol,
        json.dumps(params, sort_keys=True),
        covar_set_id,
        metrics,
        metrics_final,
        forecast,
        fit_time,
        region,
        random_seed,
        future_steps,
        n_covars,
    )

    logger.info(
        "%s - estimated %s yearly-seasonality coefficients. snapshot_id: %s",
        symbol,
        n_yearly_seasonality,
        snapshot_id,
    )
