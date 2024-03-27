import os
import sys
import logging
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


def init():
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
    logger.setLevel(logging.ERROR)

    file_handler = logging.FileHandler("etl.log")
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


def train(df, epochs=None, **kwargs):
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


def save_covar_metrics(anchor_symbol, cov_table, cov_symbol, cov_metrics):
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
        output = train(
            merged_df,
            weekly_seasonality=False,
            daily_seasonality=False,
            impute_missing=True,
        )
        # extract the last row of output, add symbol column, and consolidate to another dataframe
        last_row = output.iloc[[-1]]
        save_covar_metrics(anchor_symbol, cov_table, cov_symbol, last_row)
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
                logger.error(e)

    return results


def load_topn_covar_symbols(
    n, anchor_symbol, cov_table="index_daily_em_view", feature="change_rate"
):
    query = """
        select
            cov_symbol
        from
            neuralprophet_corel
        where
            symbol = :anchor_symbol
            and cov_table = :cov_table
            and feature = :feature
        order by
            loss_val asc
        limit :limit;
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
    )['symbol'].tolist()

    return cov_symbols

def augment_anchor_df_with_covars(df, top_n):
    merged_df = df[["ds", "y"]]
    cov100_symbols = load_topn_covar_symbols(top_n, merged_df)

    query = """
        SELECT symbol ID, date DS, change_rate y
        FROM index_daily_em_view 
        where symbol in %(symbols)s
        order by ID, DS asc
    """
    params = {'symbols': tuple(cov100_symbols)}
    cov100_daily_df = pd.read_sql(query, alchemyEngine, params=params, parse_dates=["ds"])

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

def init_search_grid():
    # Define your hyperparameters grid
    param_grid = {
        "n_lags": [None].append(list(range(2, 21))),
        "yearly_seasonality": [None].append(list(range(10, 21))),
        "ar_layers": [None].append([[i] * i for i in range(2, 17)]),
        "lagged_reg_layers": [None].append([[i] * i for i in range(2, 17)]),
    }
    grid = ParameterGrid(param_grid)
    logger.info("size of grid: %d", len(grid))
    return grid


def log_metrics_for_hyper_params(df, params):
    metrics = train(
        df,
        epochs=500,
        n_lags=params["n_lags"],
        yearly_seasonality=params["yearly_seasonality"],
        ar_layers=params["ar_layers"],
        lagged_reg_layers=params["lagged_reg_layers"],
        weekly_seasonality=False,
        daily_seasonality=False,
        impute_missing=True,
    )
    last_metric = metrics.iloc[-1]
    with alchemyEngine.begin() as conn:
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
                "model": 'NeuralProphet',
                "hyper_params": params.tostring(),
                "tag": 'baseline' if all(v is None for v in params.values()) else None,
                "mae_val": last_metric["MAE_val"],
                "rmse_val": last_metric["RMSE_val"],
                "loss_val": last_metric["Loss_val"],
                "predict_diff_mean": None,
                "predict_diff_stddev": None,
            },
        )

    return last_metric

def grid_search(df):
    grid = init_search_grid()

    # get the number of CPU cores
    num_proc = int((multiprocessing.cpu_count()) / 1.2)

    results = []
    # Use ThreadPoolExecutor to calculate metrics in parallel
    with ThreadPoolExecutor(max_workers=num_proc) as executor:
        futures = [executor.submit(log_metrics_for_hyper_params, df, params) for params in grid]
        for f in futures:
            try:
                results.append(f.result())
            except Exception as e:
                logger.error(e)


def main(args):
    init()

    anchor_symbol = args["symbol"]
    anchor_df = load_anchor_ts(anchor_symbol)

    min_date = anchor_df["ds"].min().strftime("%Y-%m-%d")
    cov_symbols = covar_symbols_index(anchor_symbol, min_date)

    if not cov_symbols.empty:
        baseline_metrics_index(anchor_symbol, anchor_df, cov_symbols)

    load_topn_covar_symbols

# Command-line argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Identify potential covariates and perform grid-search for hyper-parameters."
    )
    # Add arguments based on the requirements of the notebook code
    parser.add_argument("--covar-only", action='store_true', help="Description of arg1")
    parser.add_argument("--grid-search-only", type=str, required=True, help="Description of arg2")
    parser.add_argument("symbol", type=str, help="The asset symbol to handle.")

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args)
