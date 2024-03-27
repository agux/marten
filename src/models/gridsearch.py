import os
import sys
import logging
import json
import argparse
import multiprocessing
import pandas as pd
import numpy as np
import sqlalchemy
import exchange_calendars as xcals
from dotenv import load_dotenv

# import exchange_calendars as xcals
from datetime import datetime, timedelta

# import pytz
# import pandas as pd
# from IPython.display import display, HTML
from sqlalchemy import create_engine, text
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
    load_dotenv()  # take environment variables from .env.

    module_path = os.getenv("LOCAL_AKSHARE_DEV_MODULE")
    if module_path is not None and module_path not in sys.path:
        sys.path.insert(0, module_path)
    import akshare as ak  # noqa: E402

    print(ak.__version__)

    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT")
    DB_NAME = os.getenv("DB_NAME")

    # Create an engine instance
    alchemyEngine = create_engine(
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
        pool_recycle=3600,
    )

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler("grid_search.log", mode='w')
    console_handler = logging.StreamHandler()

    # Step 4: Create a formatter
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")

    # Step 5: Attach the formatter to the handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Step 6: Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    xshg = xcals.get_calendar("XSHG")


def _train(df, epochs=None, **kwargs):
    global alchemyEngine, logger, random_seed

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


def load_anchor_ts(symbol="930955"):
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


def baseline_metrics_index(anchor_symbol, anchor_df, covar_symbols):
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
            weekly_seasonality=False,
            daily_seasonality=False,
            impute_missing=True,
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


def _init_search_grid():
    global alchemyEngine, logger, random_seed

    # Define your hyperparameters grid
    param_grid = {
        "n_lags": list(range(0, 21)),
        "yearly_seasonality": ['auto'] + list(range(1, 21)),
        "ar_layers": [[]] + [[i] * i for i in range(2, 17)],
        "lagged_reg_layers": [[]] + [[i] * i for i in range(2, 17)],
    }
    grid = ParameterGrid(param_grid)
    logger.info("size of grid: %d", len(grid))
    return grid


def _log_metrics_for_hyper_params(df, params, epochs):
    global alchemyEngine, logger, random_seed

    # check if the params combination already exists in grid_search_metrics table. And if such, return immediately.
    query = """
        SELECT model, hyper_params
        FROM grid_search_metrics
        WHERE model = 'NeuralProphet' AND hyper_params = %(hyper_params)s
    """
    param_str = json.dumps(params)
    existing_params = pd.read_sql(
        query, alchemyEngine, params={"hyper_params": param_str}
    )
    if not existing_params.empty:
        logger.info("Skipping existing parameter combination: %s", param_str)
        return None

    metrics = _train(
        df,
        epochs=epochs,
        n_lags=params["n_lags"],
        yearly_seasonality=params["yearly_seasonality"],
        ar_layers=params["ar_layers"],
        lagged_reg_layers=params["lagged_reg_layers"],
        weekly_seasonality=False,
        daily_seasonality=False,
        impute_missing=True,
    )
    last_metric = metrics.iloc[-1]
    logger.info('%s\nparams:%s', last_metric, params)
    with alchemyEngine.begin() as conn:
        isBaseline = (
            params["n_lags"] == 0
            and params["yearly_seasonality"] == "auto"
            and params["ar_layers"] == []
            and params["lagged_reg_layers"] == []
        )
        conn.execute(
            text(
                """
                INSERT INTO grid_search_metrics 
                (model, hyper_params, tag, mae_val, rmse_val, loss_val, predict_diff_mean, predict_diff_stddev) 
                VALUES (:model, :hyper_params, :tag, :mae_val, :rmse_val, :loss_val, :predict_diff_mean, :predict_diff_stddev) 
                ON CONFLICT (model, hyper_params) 
                DO UPDATE SET mae_val = EXCLUDED.mae_val, rmse_val = EXCLUDED.rmse_val, loss_val = EXCLUDED.loss_val, predict_diff_mean = EXCLUDED.predict_diff_mean, predict_diff_stddev = EXCLUDED.predict_diff_stddev
            """
            ),
            {
                "model": "NeuralProphet",
                "hyper_params": param_str,
                "tag": "baseline" if isBaseline else None,
                "mae_val": last_metric["MAE_val"],
                "rmse_val": last_metric["RMSE_val"],
                "loss_val": last_metric["Loss_val"],
                "predict_diff_mean": None,
                "predict_diff_stddev": None,
            },
        )

    return last_metric


def grid_search(df, args):
    global alchemyEngine, logger, random_seed

    grid = _init_search_grid()

    # get the number of CPU cores
    num_proc = args.worker if args.worker is not None else int((multiprocessing.cpu_count()) / 1.2)

    results = []
    # Use ThreadPoolExecutor to calculate metrics in parallel
    with ThreadPoolExecutor(max_workers=num_proc) as executor:
        futures = [
            executor.submit(_log_metrics_for_hyper_params, df, params, args.epochs)
            for params in grid
        ]
        for f in futures:
            try:
                results.append(f.result())
            except Exception as e:
                logger.exception(e)


def main(args):
    init()

    anchor_symbol = args.symbol
    anchor_df = load_anchor_ts(anchor_symbol)

    min_date = anchor_df["ds"].min().strftime("%Y-%m-%d")
    cov_symbols = covar_symbols_index(anchor_symbol, min_date)

    if not cov_symbols.empty:
        baseline_metrics_index(anchor_symbol, anchor_df, cov_symbols)

    df = augment_anchor_df_with_covars(anchor_symbol, anchor_df, args.top_n)

    grid_search(df, args)


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
    parser.add_argument(
        "--top_n",
        action="store",
        type=int,
        default=None,
        help="Use top-n covariates for training and prediction.",
    )
    parser.add_argument(
        "--epochs",
        action="store",
        type=int,
        default=None,
        help="Epochs for training the model",
    )
    parser.add_argument(
        "--worker",
        action="store",
        type=int,
        default=None,
        help="Epochs for training the model",
    )
    parser.add_argument("symbol", type=str, help="The asset symbol to handle.")

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args)
