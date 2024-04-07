import os
import sys
import time
import logging
import json
import argparse
import hashlib
import multiprocessing
import pandas as pd
import numpy as np
import exchange_calendars as xcals
from dotenv import load_dotenv
from joblib import Parallel, delayed

# import exchange_calendars as xcals
from datetime import datetime, timedelta

from tenacity import (
    # retry,
    stop_after_attempt,
    wait_exponential,
    # retry_if_exception_type,
    Retrying,
)

# import pytz
# import pandas as pd
# from IPython.display import display, HTML
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy.dialects.postgresql import insert
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from neuralprophet import NeuralProphet, set_log_level, set_random_seed

from sklearn.model_selection import ParameterGrid

random_seed = 7
logger = None
alchemyEngine = None
args = None


def init():
    global alchemyEngine, logger, random_seed

    alchemyEngine, logger = _init_worker_resource()
    xshg = xcals.get_calendar("XSHG")


def _init_worker_resource():
    # NeuralProphet: Disable logging messages unless there is an error
    set_log_level("ERROR")

    load_dotenv()  # take environment variables from .env.

    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT")
    DB_NAME = os.getenv("DB_NAME")

    # Create an engine instance
    alchemyEngine = create_engine(
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
        # pool_recycle=3600,
        # pool_size=1,
        poolclass=NullPool,
    )
    sessionmaker(alchemyEngine)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler("grid_search.log")
    console_handler = logging.StreamHandler()

    # Step 4: Create a formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Step 5: Attach the formatter to the handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Step 6: Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return alchemyEngine, logger


def _train(df, epochs=None, random_seed=7, early_stopping=True, **kwargs):
    set_random_seed(random_seed)
    m = NeuralProphet(**kwargs)
    covars = [col for col in df.columns if col not in ("ds", "y")]
    m.add_lagged_regressor(covars)
    train_df, test_df = m.split_df(
        df,
        valid_p=1.0 / 10,
    )
    try:
        metrics = m.fit(
            train_df,
            validation_df=test_df,
            progress=None,
            epochs=epochs,
            early_stopping=early_stopping,
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


def load_anchor_ts(symbol, limit):
    global alchemyEngine, logger, random_seed
    ## support arbitrary types of symbol (could be from different tables, with different features available)
    tbl_cols_dict = {
        "index_daily_em_view": "date DS, change_rate y, vol_change_rate vol_cr, amt_change_rate amt_cr, open, close, high, low, volume, amount",
        "fund_etf_daily_em_view": "date DS, change_rate y, vol_change_rate vol_cr, amt_change_rate amt_cr, open, close, high, low, volume, turnover, turnover_rate",
        "us_index_daily_sina_view": "date DS, change_rate y, amt_change_rate amt_cr, open, close, high, low, volume, amount",
    }
    # lookup which table the symbol's data is in
    anchor_table = (
        "index_daily_em_view"  # Default table, replace with actual logic if necessary
    )
    with alchemyEngine.connect() as conn:
        result = conn.execute(
            text("""SELECT "table" FROM symbol_dict WHERE symbol = :symbol"""),
            {"symbol": symbol},
        ).fetchone()
    anchor_table = result[0] if result else anchor_table

    # load anchor TS
    query = f"""
        SELECT {tbl_cols_dict[anchor_table]}
        FROM {anchor_table}
        where symbol = %(symbol)s
        order by DS desc
    """
    params = {"symbol": symbol}
    if limit > 0:
        query += " limit %(limit)s"
        params["limit"] = limit
    df = pd.read_sql(
        query,
        alchemyEngine,
        params=params,
        parse_dates=["ds"],
    )
    # re-order rows by ds (which is a date column) descending (most recent date in the top row)
    df.sort_values(by="ds", ascending=False, inplace=True)
    return df, anchor_table


def _covar_symbols_from_table(anchor_symbol, min_date, table, feature):
    global alchemyEngine, logger, random_seed
    # get a list of other China indices, and not yet have metrics recorded
    query = f"""
        select
            distinct t.symbol
        from
            {table} t
        where
            t.symbol <> %(anchor_symbol)s
            and t.date <= %(min_date)s
            and t.symbol not in (
                select
                    cov_symbol
                from
                    neuralprophet_corel nc
                where
                    symbol = %(anchor_symbol)s
                    and cov_table = %(table)s
                    and feature = %(feature)s
            )
    """
    params = {
        "table": table,
        "anchor_symbol": anchor_symbol,
        "feature": feature,
        "min_date": min_date,
    }
    cov_symbols = pd.read_sql(query, alchemyEngine, params=params)
    return cov_symbols


def _save_covar_metrics(
    anchor_symbol,
    cov_table,
    cov_symbol,
    feature,
    cov_metrics,
    fit_time,
    timesteps,
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
                    (symbol, cov_table, cov_symbol, feature, mae_val, rmse_val, loss_val, fit_time, timesteps) 
                    VALUES (:symbol, :cov_table, :cov_symbol, :feature, :mae_val, :rmse_val, :loss_val, :fit_time, :timesteps) 
                    ON CONFLICT (symbol, cov_symbol, feature, cov_table) 
                    DO UPDATE SET 
                        mae_val = EXCLUDED.mae_val, 
                        rmse_val = EXCLUDED.rmse_val, 
                        loss_val = EXCLUDED.loss_val,
                        fit_time = EXCLUDED.fit_time,
                        timesteps = EXCLUDED.timesteps
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
                },
            )


def _fit_with_covar(
    anchor_symbol,
    anchor_df,
    cov_table,
    cov_symbol,
    min_date,
    random_seed,
    feature,
    accelerator,
):
    alchemyEngine, logger = _init_worker_resource()
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
        metrics = _train(
            df=merged_df,
            epochs=None,
            random_seed=random_seed,
            early_stopping=True,
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
    _save_covar_metrics(
        anchor_symbol,
        cov_table,
        cov_symbol,
        feature,
        last_row,
        fit_time,
        timesteps,
        alchemyEngine,
    )
    return last_row


def _pair_endogenous_covar_metrics(anchor_symbol, anchor_df, cov_table, features, args):
    global random_seed

    # remove feature elements already exists in the neuralprophet_corel table.
    features = _remove_measured_features(anchor_symbol, cov_table, features)

    if not features:
        return

    # get the number of CPU cores
    n_jobs = (
        args.worker
        if args.worker is not None
        else int((multiprocessing.cpu_count()) / 1.5)
    )
    Parallel(n_jobs=n_jobs)(
        delayed(_fit_with_covar)(
            anchor_symbol,
            anchor_df,
            cov_table,
            anchor_symbol,
            None,
            random_seed,
            feature,
            "auto" if args.accelerator else None,
        )
        for feature in features
    )


def _pair_covar_metrics(
    anchor_symbol, anchor_df, cov_table, cov_symbols, feature, args
):
    global random_seed
    # get the number of CPU cores
    n_jobs = (
        args.worker
        if args.worker is not None
        else int((multiprocessing.cpu_count()) / 1.5)
    )
    min_date = anchor_df["ds"].min().strftime("%Y-%m-%d")
    Parallel(n_jobs=n_jobs)(
        delayed(_fit_with_covar)(
            anchor_symbol,
            anchor_df,
            cov_table,
            symbol,
            min_date,
            random_seed,
            feature,
            "auto" if args.accelerator else None,
        )
        for symbol in cov_symbols["symbol"]
    )


def _load_covar_set(covar_set_id):
    global alchemyEngine
    query = """
        select
            cov_symbol, cov_table, cov_feature
        from
            covar_set
        where
            id = %(covar_set_id)s
    """
    params = {
        "covar_set_id": covar_set_id,
    }
    df = pd.read_sql(
        query,
        alchemyEngine,
        params=params,
    )
    return df


def _load_topn_covars(n, anchor_symbol, cov_table=None, feature=None):
    global alchemyEngine
    query = """
        select
            cov_symbol, cov_table, feature
        from
            neuralprophet_corel
        where
            symbol = %(anchor_symbol)s
    """
    params = {
        "anchor_symbol": anchor_symbol,
        "limit": n,
    }
    if cov_table is not None:
        query += " and cov_table = %(cov_table)s"
        params["cov_table"] = cov_table
    if feature is not None:
        query += " and feature = %(feature)s"
        params["feature"] = feature

    query += " order by loss_val asc limit %(limit)s"
    df = pd.read_sql(
        query,
        alchemyEngine,
        params=params,
    )

    # get next sequence value from covar_set_sequence.
    with alchemyEngine.begin() as conn:
        covar_set_id = conn.execute(
            text("SELECT nextval('covar_set_sequence')")
        ).scalar()

    return df, covar_set_id


def augment_anchor_df_with_covars(df, args):
    global alchemyEngine, logger
    merged_df = df[["ds", "y"]]
    if args.covar_set_id is not None:
        # TODO load covars based on the set id
        covar_set_id = args.covar_set_id
        covars_df = _load_covar_set(covar_set_id)
    else:
        covars_df, covar_set_id = _load_topn_covars(args.top_n, args.symbol)

    logger.info("loaded top %s covariates", len(covars_df))

    # covars_df contain these columns: cov_symbol, cov_table, feature
    by_table_feature = covars_df.groupby(["cov_table", "feature"])
    for group1, _ in by_table_feature:
        ## TODO need to load covariate time series from different tables and/or features
        cov_table = group1[0]
        feature = group1[1]

        query = f"""
            SELECT symbol ID, date DS, {feature} y
            FROM {cov_table}
            where symbol in %(symbols)s
            order by ID, DS asc
        """
        params = {
            "symbols": tuple(covars_df["cov_symbol"]),
        }
        cov_daily_df = pd.read_sql(
            query, alchemyEngine, params=params, parse_dates=["ds"]
        )

        # merge and append the feature column of cov_daily_df to merged_df, by matching dates
        # split cov_daily_df by symbol column
        grouped = cov_daily_df.groupby("id")
        for group2, sdf in grouped:
            col_name = f"{feature}_{group2}"
            sdf = sdf.rename(
                columns={
                    "y": col_name,
                }
            )
            sdf = sdf[["ds", col_name]]
            merged_df = pd.merge(merged_df, sdf, on="ds", how="left")

    return merged_df, covar_set_id


def _get_layers():
    layers = []
    # Loop over powers of 2 from 2^1 to 2^6
    for i in range(1, 7):
        power_of_two = 2**i
        # Loop over list lengths from 2 to 16
        for j in range(2, 17):
            # Create a list with the current power of two, repeated 'j' times
            element = [power_of_two] * j
            # Append the list to the result
            layers.append(element)
    return layers


def _init_search_grid():
    global alchemyEngine, logger, random_seed

    layers = _get_layers()

    # Define your hyperparameters grid
    param_grid = {
        {
            # default hyperparameters
            "batch_size": None,
            "n_lags": 0,
            "yearly_seasonality": "auto",
            "ar_layers": [],
            "lagged_reg_layers": [],
        },
        {
            "batch_size": [None, 50, 100, 200],
            "n_lags": list(range(1, 31)),
            "yearly_seasonality": list(range(5, 30)),
            "ar_layers": layers,
            "lagged_reg_layers": layers,
        },
    }
    grid = ParameterGrid(param_grid)
    logger.info("size of grid: %d", len(grid))
    return grid


# @retry(
#     stop=stop_after_attempt(3),
#     wait=wait_exponential(multiplier=1, max=5),
# )
def _new_metric_keys(anchor_symbol, hpid, hyper_params, covar_set_id, alchemyEngine):
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


def _update_metrics_table(
    alchemyEngine, params, anchor_symbol, hpid, epochs, last_metric, fit_time
):
    def action():
        with alchemyEngine.begin() as conn:
            tag = (
                "baseline,multivariate"
                if (
                    params["batch_size"] is None
                    and params["n_lags"] == 0
                    and params["yearly_seasonality"] == "auto"
                    and params["ar_layers"] == []
                    and params["lagged_reg_layers"] == []
                )
                else None
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
                """
                ),
                {
                    "model": "NeuralProphet",
                    "anchor_symbol": anchor_symbol,
                    "hpid": hpid,
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


def _log_metrics_for_hyper_params(
    anchor_symbol, df, params, epochs, random_seed, accelerator, covar_set_id
):
    alchemyEngine, logger = _init_worker_resource()

    # to support distributed processing, we try to insert a new record (with primary keys only)
    # into grid_search_metrics first. If we hit duplicated key error, return None.
    # Otherwise we could proceed further code execution.
    param_str = json.dumps(params)
    hpid = hashlib.md5(param_str.encode("utf-8")).hexdigest()
    if not _new_metric_keys(
        anchor_symbol, hpid, param_str, covar_set_id, alchemyEngine
    ):
        logger.debug("Skip re-entry for %s: %s", anchor_symbol, param_str)
        return None

    start_time = time.time()
    metrics = None
    try:
        metrics = _train(
            df,
            epochs=epochs,
            random_seed=random_seed,
            early_stopping=True,
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
        return None

    fit_time = time.time() - start_time
    last_metric = metrics.iloc[-1]
    covars = [col for col in df.columns if col not in ("ds", "y")]
    logger.info("%s\nparams:%s\n#covars:%s", last_metric, params, len(covars))

    _update_metrics_table(
        alchemyEngine, params, anchor_symbol, hpid, last_metric['epoch']+1, last_metric, fit_time
    )

    return last_metric


def grid_search(df, covar_set_id, args):
    global alchemyEngine, logger, random_seed

    grid = _init_search_grid()

    # get the number of CPU cores
    n_jobs = (
        args.worker
        if args.worker is not None
        else int((multiprocessing.cpu_count()) / 1.2)
    )

    Parallel(n_jobs=n_jobs)(
        delayed(_log_metrics_for_hyper_params)(
            args.symbol,
            df,
            params,
            args.epochs,
            random_seed,
            "auto" if args.accelerator else None,
            covar_set_id,
        )
        for params in grid
    )


def _remove_measured_features(anchor_symbol, cov_table, features):
    query = f"""
        select feature
        from neuralprophet_corel
        where symbol = %(symbol)s
        and cov_table = %(cov_table)s
        and feature in %(features)s
    """
    existing_features_pd = pd.read_sql(
        query,
        alchemyEngine,
        params={
            "symbol": anchor_symbol,
            "cov_table": cov_table,
            "features": tuple(features),
        },
    )
    # remove elements in the `features` list that exist in the existing_features_pd.
    features = list(set(features) - set(existing_features_pd["feature"]))
    return features


def _covar_metric(anchor_symbol, anchor_df, cov_table, features, min_date, args):
    features = _remove_measured_features(anchor_symbol, cov_table, features)
    for feature in features:
        if cov_table != "bond_metrics_em":
            cov_symbols = _covar_symbols_from_table(
                anchor_symbol, min_date, cov_table, feature
            )
            # remove duplicate records in cov_symbols dataframe, by checking the `symbol` column values.
            cov_symbols.drop_duplicates(subset=["symbol"], inplace=True)
        else:
            # construct a dummy cov_symbols dataframe with `symbol` column and the value 'bond'.
            cov_symbols = pd.DataFrame({"symbol": ["bond"]})
        if not cov_symbols.empty and features:
            _pair_covar_metrics(
                anchor_symbol,
                anchor_df,
                cov_table,
                cov_symbols,
                feature,
                args,
            )


def prep_covar_baseline_metrics(anchor_df, anchor_table, args):
    global random_seed

    anchor_symbol = args.symbol
    min_date = anchor_df["ds"].min().strftime("%Y-%m-%d")

    # endogenous features of the anchor time series per se
    endogenous_features = [col for col in anchor_df.columns if col not in ("ds")]
    _pair_endogenous_covar_metrics(
        anchor_symbol, anchor_df, anchor_table, endogenous_features, args
    )

    # for the rest of exogenous covariates, keep only the core features of anchor_df
    anchor_df = anchor_df[["ds", "y"]]

    # prep CN index covariates
    features = ["change_rate", "amt_change_rate"]
    cov_table = "index_daily_em_view"
    _covar_metric(anchor_symbol, anchor_df, cov_table, features, min_date, args)

    # prep ETF covariates  fund_etf_daily_em_view
    features = ["change_rate", "amt_change_rate"]
    cov_table = "fund_etf_daily_em_view"
    _covar_metric(anchor_symbol, anchor_df, cov_table, features, min_date, args)

    # prep bond covariates bond_metrics_em
    features = [
        "china_yield_2y",
        "china_yield_10y",
        "china_yield_30y",
        "china_yield_spread_10y_2y",
        "us_yield_2y",
        "us_yield_10y",
        "us_yield_30y",
        "us_yield_spread_10y_2y",
    ]
    cov_table = "bond_metrics_em"
    _covar_metric(anchor_symbol, anchor_df, cov_table, features, min_date, args)

    # prep US index covariates us_index_daily_sina
    features = ["change_rate", "amt_change_rate"]
    cov_table = "us_index_daily_sina_view"
    _covar_metric(anchor_symbol, anchor_df, cov_table, features, min_date, args)

    # prep HK index covariates hk_index_daily_sina
    features = ["change_rate"]
    cov_table = "hk_index_daily_em_view"
    _covar_metric(anchor_symbol, anchor_df, cov_table, features, min_date, args)

    # TODO: prep stock features

    # TODO prep options
    # TODO RMB exchange rate
    # TODO CPI, PPI
    # TODO car sales
    # TODO electricity consumption
    # TODO exports and imports
    # TODO commodity prices: oil, copper, aluminum, coal, gold, etc.
    # TODO cash inflow


def main(args):
    init()

    anchor_df, anchor_table = load_anchor_ts(args.symbol, args.timestep_limit)

    if not args.grid_search_only:
        prep_covar_baseline_metrics(anchor_df, anchor_table, args)

    if not args.covar_only:
        df, covar_set_id = augment_anchor_df_with_covars(anchor_df, args)
        grid_search(df, covar_set_id, args)


# Command-line argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Identify potential covariates and perform grid-search for hyper-parameters."
    )

    # Create a mutually exclusive group
    group1 = parser.add_mutually_exclusive_group(required=False)
    # Add arguments based on the requirements of the notebook code
    group1.add_argument(
        "--covar_only",
        action="store_true",
        help="Collect paired covariate metrics in neuralprophet_corel table only.",
    )
    group1.add_argument(
        "--grid_search_only", action="store_true", help="Perform grid search only."
    )

    # Create a mutually exclusive group
    group2 = parser.add_mutually_exclusive_group(required=False)
    # Add arguments based on the requirements of the notebook code
    group2.add_argument(
        "--top_n",
        action="store",
        type=int,
        default=100,
        help="Use top-n covariates for training and prediction.",
    )
    group2.add_argument(
        "--covar_set_id",
        action="store",
        type=int,
        default=None,
        help=(
            "Covariate set ID corresponding to the covar_set table. ",
            "If not set, the grid search will look for latest top_n covariates ",
            "as found in the neuralprophet_corel table, which could be non-static.",
        ),
    )

    parser.add_argument(
        "--epochs",
        action="store",
        type=int,
        default=500,
        help="Epochs for training the model",
    )
    parser.add_argument(
        "--worker",
        action="store",
        type=int,
        default=-1,
        help="Number or parallel workers (python processes) for training the model",
    )
    parser.add_argument(
        "--timestep_limit",
        action="store",
        type=int,
        default=1200,
        help="Limit the time steps of anchor symbol to the most recent N data points. Specify -1 to utilize all time steps available.",
    )
    parser.add_argument(
        "--accelerator", action="store_true", help="Use accelerator automatically"
    )

    parser.add_argument(
        "symbol", type=str, help="The asset symbol as anchor to be analyzed."
    )

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    try:
        main(args)
    except Exception as e:
        logger.exception("encountered exception in main()")
