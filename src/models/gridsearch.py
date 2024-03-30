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

# Disable logging messages unless there is an error
set_log_level("ERROR")

random_seed = 7
logger = None
alchemyEngine = None
args = None


def init():
    global alchemyEngine, logger, random_seed

    alchemyEngine, logger = _init_worker_resource()
    xshg = xcals.get_calendar("XSHG")


def _init_worker_resource():
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
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")

    # Step 5: Attach the formatter to the handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Step 6: Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return alchemyEngine, logger


def _train(df, epochs=None, random_seed=7, **kwargs):
    set_random_seed(random_seed)
    m = NeuralProphet(**kwargs)
    covars = [col for col in df.columns if col not in ("ds", "y")]
    m.add_lagged_regressor(covars)
    train_df, test_df = m.split_df(
        df,
        valid_p=1.0 / 10,
    )
    metrics = m.fit(train_df, validation_df=test_df, progress=None, epochs=epochs)
    return metrics


def load_anchor_ts(symbol):
    global alchemyEngine, logger, random_seed
    # load anchor TS
    query = f"""
        SELECT date DS, change_rate y, vol_change_rate vol_cr, amt_change_rate amt_cr, 
            open, close, high, low, volume, amount
        FROM index_daily_em_view
        where symbol='{symbol}'
        order by DS
    """
    df = pd.read_sql(query, alchemyEngine, parse_dates=["ds"])
    return df


def covar_symbols_index(anchor_symbol, min_date):
    global alchemyEngine, logger, random_seed
    # get a list of other China indices, and not yet have metrics recorded
    query = f"""
        select
            distinct idev.symbol
        from
            index_daily_em_view idev
        inner join neuralprophet_corel nc on
            (
                idev.symbol = nc.symbol
            )
        where
            idev.symbol <> '{anchor_symbol}'
            and idev.symbol not in (
                select
                    cov_symbol
                from
                    neuralprophet_corel nc
                where
                    symbol = '{anchor_symbol}'
                    and cov_table = 'index_daily_em_view'
                    and feature = 'change_rate'
            )
            and date <= '{min_date}';
    """
    cov_symbols = pd.read_sql(query, alchemyEngine)
    return cov_symbols


def _save_covar_metrics(anchor_symbol, cov_table, cov_symbol, cov_metrics):
    global alchemyEngine, logger, random_seed
    # Insert data into the table
    with alchemyEngine.begin() as conn:
        # Inserting DataFrame into the database table
        for index, row in cov_metrics.iterrows():
            conn.execute(
                text(
                    """
                    INSERT INTO neuralprophet_corel 
                    (symbol, cov_table, cov_symbol, feature, mae_val, rmse_val, loss_val) 
                    VALUES (:symbol, :cov_table, :cov_symbol, :feature, :mae_val, :rmse_val, :loss_val) 
                    ON CONFLICT (symbol, cov_symbol, feature) 
                    DO UPDATE SET mae_val = EXCLUDED.mae_val, rmse_val = EXCLUDED.rmse_val, loss_val = EXCLUDED.loss_val
                """
                ),
                {
                    "symbol": anchor_symbol,
                    "cov_table": cov_table,
                    "cov_symbol": cov_symbol,
                    "feature": "change_rate",
                    "mae_val": row["MAE_val"],
                    "rmse_val": row["RMSE_val"],
                    "loss_val": row["Loss_val"],
                },
            )


def baseline_metrics_index(anchor_symbol, anchor_df, covar_symbols, batch_size=None, accelerator=None):
    global alchemyEngine, logger, random_seed

    min_date = anchor_df["ds"].min().strftime("%Y-%m-%d")
    cov_table = "index_daily_em_view"

    def load_train(cov_symbol):
        query = f"""
            select date ds, change_rate y_{cov_symbol}
            from {cov_table}
            where symbol = '{cov_symbol}'
            and date >= '{min_date}'
            order by date
        """
        cov_symbol_df = pd.read_sql(query, alchemyEngine)
        if cov_symbol_df.empty:
            return None
        merged_df = pd.merge(anchor_df, cov_symbol_df, on="ds", how="left")
        output = _train(
            merged_df,
            random_seed=random_seed,
            batch_size=batch_size,
            weekly_seasonality=False,
            daily_seasonality=False,
            impute_missing=True,
            accelerator=accelerator,
        )
        # extract the last row of output, add symbol column, and consolidate to another dataframe
        last_row = output.iloc[[-1]]
        _save_covar_metrics(anchor_symbol, cov_table, cov_symbol, last_row)
        return last_row

    # get the number of CPU cores
    num_proc = int((multiprocessing.cpu_count() + 1) / 1.5)

    results = []
    # Use ThreadPoolExecutor to calculate metrics in parallel
    with ThreadPoolExecutor(max_workers=num_proc) as executor:
        futures = [
            executor.submit(load_train, symbol) for symbol in covar_symbols["symbol"]
        ]
        for f in futures:
            try:
                results.append(f.result())
            except Exception as e:
                logger.exception(e)

    return results


def _load_topn_covar_symbols(
    n, anchor_symbol, cov_table="index_daily_em_view", feature="change_rate"
):
    global alchemyEngine, logger, random_seed
    query = """
        select
            cov_symbol
        from
            neuralprophet_corel
        where
            symbol = %(anchor_symbol)s
            and cov_table = %(cov_table)s
            and feature = %(feature)s
        order by
            loss_val asc
        limit %(limit)s
    """
    cov_symbols = pd.read_sql(
        query,
        alchemyEngine,
        params={
            "anchor_symbol": anchor_symbol,
            "cov_table": cov_table,
            "feature": feature,
            "limit": n,
        },
    )["cov_symbol"].tolist()

    return cov_symbols


def augment_anchor_df_with_covars(anchor_symbol, df, top_n=100):
    global alchemyEngine, logger, random_seed
    merged_df = df[["ds", "y"]]
    cov100_symbols = _load_topn_covar_symbols(top_n, anchor_symbol)
    logger.info("loaded top %s covariates", len(cov100_symbols))
    query = """
        SELECT symbol ID, date DS, change_rate y
        FROM index_daily_em_view 
        where symbol in %(symbols)s
        order by ID, DS asc
    """
    params = {"symbols": tuple(cov100_symbols)}
    cov100_daily_df = pd.read_sql(
        query, alchemyEngine, params=params, parse_dates=["ds"]
    )

    # merge and append the 'change_rate' column of cov_df to df, by matching dates
    # split cov_df by symbol column
    grouped = cov100_daily_df.groupby("id")
    # sub_dfs = {group: data for group, data in grouped}
    for group, sdf in grouped:
        sdf = sdf.rename(
            columns={
                "y": f"y_{group}",
            }
        )
        sdf = sdf[["ds", f"y_{group}"]]
        merged_df = pd.merge(merged_df, sdf[["ds", f"y_{group}"]], on="ds", how="left")

    return merged_df


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
        "batch_size": [None, 50, 100, 200],
        "n_lags": list(range(0, 31)),
        "yearly_seasonality": ["auto"] + list(range(1, 25)),
        "ar_layers": layers,
        "lagged_reg_layers": layers,
    }
    grid = ParameterGrid(param_grid)
    logger.info("size of grid: %d", len(grid))
    return grid


# @retry(
#     stop=stop_after_attempt(3),
#     wait=wait_exponential(multiplier=1, max=5),
# )
def _new_metric_keys(anchor_symbol, hpid, hyper_params, alchemyEngine):
    def action():
        try:
            with alchemyEngine.begin() as conn:
                conn.execute(
                    text(
                        """
                        INSERT INTO grid_search_metrics (model, anchor_symbol, hpid, hyper_params) 
                        VALUES (:model, :anchor_symbol, :hpid, :hyper_params)
                        """
                    ),
                    {
                        "model": "NeuralProphet",
                        "anchor_symbol": anchor_symbol,
                        "hpid": hpid,
                        "hyper_params": hyper_params,
                    },
                )
                return True
        except Exception as e:
            if "duplicate key value violates unique constraint" in str(e):
                return False
            else:
                raise

    for attempt in Retrying(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=5)):
        with attempt:
            return action()


# @retry(
#     stop=stop_after_attempt(3),
#     wait=wait_exponential(multiplier=1, max=5),
# )
def _update_metrics_table(
    alchemyEngine, params, anchor_symbol, hpid, epochs, last_metric, execution_time
):
    def action():
        with alchemyEngine.begin() as conn:
            isBaseline = (
                params["batch_size"] is None
                and params["n_lags"] == 0
                and params["yearly_seasonality"] == "auto"
                and params["ar_layers"] == []
                and params["lagged_reg_layers"] == []
            )
            conn.execute(
                text(
                    """
                    UPDATE grid_search_metrics
                    SET 
                        mae_val = :mae_val, 
                        rmse_val = :rmse_val, 
                        loss_val = :loss_val, 
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
                    "tag": "baseline" if isBaseline else None,
                    "mae_val": last_metric["MAE_val"],
                    "rmse_val": last_metric["RMSE_val"],
                    "loss_val": last_metric["Loss_val"],
                    "fit_time": (str(execution_time) + " seconds",),
                    "epochs": epochs,
                },
            )

    for attempt in Retrying(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=5)
    ):
        with attempt:
            action()


def _log_metrics_for_hyper_params(anchor_symbol, df, params, epochs, random_seed, accelerator):
    alchemyEngine, logger = _init_worker_resource()

    # to support distributed processing, we try to insert a new record (with primary keys only)
    # into grid_search_metrics first. If we hit duplicated key error, return None.
    # Otherwise we could proceed further code execution.
    param_str = json.dumps(params)
    hpid = hashlib.md5(param_str.encode("utf-8")).hexdigest()
    if not _new_metric_keys(anchor_symbol, hpid, param_str, alchemyEngine):
        logger.debug("Skip re-entry for %s: %s", anchor_symbol, param_str)
        return None

    start_time = time.time()
    metrics = _train(
        df,
        epochs=epochs,
        random_seed=random_seed,
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
    execution_time = time.time() - start_time
    last_metric = metrics.iloc[-1]
    logger.info("%s\nparams:%s", last_metric, params)

    _update_metrics_table(
        alchemyEngine, params, anchor_symbol, hpid, epochs, last_metric, execution_time
    )

    return last_metric


def grid_search(anchor_symbol, df, args):
    global alchemyEngine, logger, random_seed

    grid = _init_search_grid()

    # get the number of CPU cores
    n_jobs = (
        args.worker
        if args.worker is not None
        else int((multiprocessing.cpu_count()) / 1.2)
    )

    results = Parallel(n_jobs=n_jobs)(
        delayed(_log_metrics_for_hyper_params)(
            anchor_symbol,
            df,
            params,
            args.epochs,
            random_seed,
            "auto" if args.accelerator else None
        )
        for params in grid
    )


def main(args):
    init()

    anchor_symbol = args.symbol
    anchor_df = load_anchor_ts(anchor_symbol)

    min_date = anchor_df["ds"].min().strftime("%Y-%m-%d")
    cov_symbols = covar_symbols_index(anchor_symbol, min_date)

    if not cov_symbols.empty:
        baseline_metrics_index(anchor_symbol, anchor_df, cov_symbols, accelerator="auto" if args.accelerator else None)

    df = augment_anchor_df_with_covars(anchor_symbol, anchor_df, args.top_n)

    grid_search(anchor_symbol, df, args)


# Command-line argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Identify potential covariates and perform grid-search for hyper-parameters."
    )
    # Add arguments based on the requirements of the notebook code
    parser.add_argument("--covar_only", action="store_true", help="Description of arg1")
    parser.add_argument(
        "--grid_search_only", action="store_true", help="Description of arg2"
    )
    parser.add_argument("--accelerator", action="store_true", help="Use accelerator automatically")
    parser.add_argument(
        "--top_n",
        action="store",
        type=int,
        default=100,
        help="Use top-n covariates for training and prediction.",
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
        help="Epochs for training the model",
    )
    parser.add_argument("symbol", type=str, help="The asset symbol to handle.")

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    try:
        main(args)
    except Exception as e:
        logger.error("encountered exception in main():", e)
