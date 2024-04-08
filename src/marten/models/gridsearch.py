import os
import time
import argparse
import pandas as pd
import multiprocessing
import exchange_calendars as xcals
from dotenv import load_dotenv
from dask.distributed import Client, as_completed

from marten.utils.database import get_database_engine
from marten.utils.logger import get_logger
from marten.models.worker import LocalWorkerPlugin
from marten.models.worker_func import fit_with_covar, log_metrics_for_hyper_params

from sqlalchemy import text

from neuralprophet import set_log_level

from sklearn.model_selection import ParameterGrid

random_seed = 7
logger = None
alchemyEngine = None
args = None
client = None


def init(args):
    global alchemyEngine, logger, random_seed, client

    alchemyEngine, logger = _init_worker_resource()
    xshg = xcals.get_calendar("XSHG")

    client = Client(
        n_workers=args.worker if args.worker > 0 else multiprocessing.cpu_count(),
        threads_per_worker=1,
    )
    client.register_plugin(LocalWorkerPlugin(__name__))


def _init_worker_resource():
    # NeuralProphet: Disable logging messages unless there is an error
    set_log_level("ERROR")

    load_dotenv()  # take environment variables from .env.

    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT")
    DB_NAME = os.getenv("DB_NAME")

    db_url = (
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    alchemyEngine = get_database_engine(db_url)
    logger = get_logger(__name__)

    return alchemyEngine, logger


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
    # get a list of symbols from the given table, of which metrics are not recorded yet
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


def _pair_endogenous_covar_metrics(anchor_symbol, anchor_df, cov_table, features, args):
    global random_seed, client

    # remove feature elements already exists in the neuralprophet_corel table.
    features = _remove_measured_features(anchor_symbol, cov_table, features)

    if not features:
        return

    futures = []
    for feature in features:
        future = client.submit(
            fit_with_covar,
            anchor_symbol,
            anchor_df,
            cov_table,
            anchor_symbol,
            None,
            random_seed,
            feature,
            "auto" if args.accelerator else None,
            args.early_stopping,
        )
        futures.append(future)

    # Wait for all futures to complete
    for future in as_completed(futures):
        future.result()  # We call result() to potentially raise exceptions


def _pair_covar_metrics(
    anchor_symbol, anchor_df, cov_table, cov_symbols, feature, args
):
    global random_seed, client
    min_date = anchor_df["ds"].min().strftime("%Y-%m-%d")
    futures = []
    for symbol in cov_symbols["symbol"]:
        future = client.submit(
            fit_with_covar,
            anchor_symbol,
            anchor_df,
            cov_table,
            symbol,
            min_date,
            random_seed,
            feature,
            "auto" if args.accelerator else None,
            args.early_stopping,
        )
        futures.append(future)

    # Wait for all futures to complete
    for future in as_completed(futures):
        future.result()  # Call result() to potentially raise exceptions from the tasks


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


def _load_topn_covars(n, anchor_symbol, nan_threshold=None, cov_table=None, feature=None):
    global alchemyEngine
    sub_query = """
		select
			loss_val
		from
			neuralprophet_corel
		where
			symbol = %(anchor_symbol)s
			and cov_symbol = %(anchor_symbol)s
			and feature = 'y'
    """
    query = f"""
        select
            cov_symbol, cov_table, feature
        from
            neuralprophet_corel
        where
            symbol = %(anchor_symbol)s
            and loss_val < ({sub_query})
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
    if nan_threshold is not None:
        query += " and nan_count < %(nan_threshold)s"
        params["nan_threshold"] = nan_threshold

    query += " order by loss_val asc limit %(limit)s"
    df = pd.read_sql(
        query,
        alchemyEngine,
        params=params,
    )

    if df.empty:
        return df, -1

    with alchemyEngine.begin() as conn:
        # check if the same set of covar features exists in `covar_set` table. If so, reuse the same set_id.
        query = text(
            """
            select id, count(1) num
            from covar_set
            where 
                symbol = :symbol
                and (cov_symbol, cov_table, cov_feature) IN :values
            group by id
            having count(1) = :num
            order by id
        """
        )
        params = {
            "symbol": anchor_symbol,
            "values": tuple(df.itertuples(index=False, name=None)),
            "num": len(df),
        }
        result = conn.execute(query, params)
        first_row = result.first()
        covar_set_id = None
        if first_row is not None:
            covar_set_id = first_row[0]
        else:
            # no existing set. get next sequence value from covar_set_sequence and save the set to covar_set.
            covar_set_id = conn.execute(
                text("SELECT nextval('covar_set_sequence')")
            ).scalar()
            ## insert df into covar_set table
            table_df = df.rename(columns={"feature": "cov_feature"})
            table_df["symbol"] = anchor_symbol
            table_df["id"] = covar_set_id
            table_df.to_sql("covar_set", con=conn, if_exists="append", index=False)

    return df, covar_set_id


def augment_anchor_df_with_covars(df, args):
    global alchemyEngine, logger
    merged_df = df[["ds", "y"]]
    if args.covar_set_id is not None:
        # TODO load covars based on the set id
        covar_set_id = args.covar_set_id
        covars_df = _load_covar_set(covar_set_id)
    else:
        nan_threshold = round(len(df) * args.nan_limit, 0)
        covars_df, covar_set_id = _load_topn_covars(args.top_n, args.symbol, nan_threshold)

    if covars_df.empty:
        raise Exception(f"No qualified covariates can be found for {args.symbol}. Please check the data in table neuralprophet_corel")

    logger.info(
        "loaded top %s qualified covariates. covar_set id: %s", len(covars_df), covar_set_id
    )

    # covars_df contain these columns: cov_symbol, cov_table, feature
    by_table_feature = covars_df.groupby(["cov_table", "feature"])
    for group1, sdf1 in by_table_feature:
        ## load covariate time series from different tables and/or features
        cov_table = group1[0]
        feature = group1[1]
        if cov_table != "bond_metrics_em":
            query = f"""
                SELECT symbol ID, date DS, {feature} y
                FROM {cov_table}
                where symbol in %(symbols)s
                order by ID, DS asc
            """
            params = {
                "symbols": tuple(sdf1["cov_symbol"]),
            }
            table_feature_df = pd.read_sql(
                query, alchemyEngine, params=params, parse_dates=["ds"]
            )
        else:
            query = f"""
                SELECT 'bond' ID, date DS, {feature} y
                FROM {cov_table}
                order by DS asc
            """
            table_feature_df = pd.read_sql(
                query, alchemyEngine, parse_dates=["ds"]
            )

        # merge and append the feature column of table_feature_df to merged_df, by matching dates
        # split table_feature_df by symbol column
        grouped = table_feature_df.groupby("id")
        for group2, sdf2 in grouped:
            col_name = f"{feature}_{group2}"
            sdf2.rename(
                columns={
                    "y": col_name,
                },
                inplace=True,
            )
            sdf2 = sdf2[["ds", col_name]]
            merged_df = pd.merge(merged_df, sdf2, on="ds", how="left")

    missing_values = merged_df.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    logger.info("count of missing values:\n%s", missing_values)

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
    global logger

    layers = _get_layers()

    # Define your hyperparameters grid
    param_grid = [
        {
            # default hyperparameters
            "batch_size": [None],
            "n_lags": [0],
            "yearly_seasonality": ["auto"],
            "ar_layers": [[]],
            "lagged_reg_layers": [[]],
        },
        {
            "batch_size": [None, 50, 100, 200],
            "n_lags": list(range(1, 31)),
            "yearly_seasonality": list(range(5, 30)),
            "ar_layers": layers,
            "lagged_reg_layers": layers,
        },
    ]
    grid = ParameterGrid(param_grid)
    logger.info("size of grid: %d", len(grid))
    return grid


def _cleanup_stale_keys():
    global alchemyEngine, logger
    with alchemyEngine.begin() as conn:
        conn.execute(
            text(
                """
                delete from grid_search_metrics
                where loss_val is null 
                    and last_modified <= NOW() - INTERVAL '1 hour'
                """
            )
        )


def grid_search(df, covar_set_id, args):
    global alchemyEngine, logger, random_seed, client

    _cleanup_stale_keys()

    grid = _init_search_grid()

    futures = []
    for params in grid:
        future = client.submit(
            log_metrics_for_hyper_params,
            args.symbol,
            df,
            params,
            args.epochs,
            random_seed,
            "auto" if args.accelerator else None,
            covar_set_id,
            args.early_stopping,
        )
        futures.append(future)

    # Wait for all futures to complete
    for future in as_completed(futures):
        future.result()  # Call result() to potentially raise exceptions from the tasks


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


def univariate_baseline(anchor_df, args):
    global random_seed, client
    df = anchor_df[["ds", "y"]]
    default_params = {
        # default hyperparameters. the order of keys MATTER (which affects the PK in table)
        "ar_layers": [],
        "batch_size": None,
        "lagged_reg_layers": [],
        "n_lags": 0,
        "yearly_seasonality": "auto",
    }
    client.submit(
        log_metrics_for_hyper_params,
        args.symbol,
        df,
        default_params,
        args.epochs,
        random_seed,
        "auto" if args.accelerator else None,
        0,
        args.early_stopping,
    ).result()


def main(args):
    global client, logger
    t_start = time.time()
    try:
        init(args)

        anchor_df, anchor_table = load_anchor_ts(args.symbol, args.timestep_limit)

        univariate_baseline(anchor_df, args)

        if not args.grid_search_only:
            t1_start = time.time()
            prep_covar_baseline_metrics(anchor_df, anchor_table, args)
            logger.info("%s covariate baseline metric computation completed. Time taken: %s seconds", args.symbol, time.time() - t1_start)

        if not args.covar_only:
            t2_start = time.time()
            df, covar_set_id = augment_anchor_df_with_covars(anchor_df, args)
            grid_search(df, covar_set_id, args)
            logger.info(
                "%s grid-search completed. Time taken: %s seconds",
                args.symbol,
                time.time() - t2_start,
            )
    finally:
        if client is not None:
            # Remember to close the client if your program is done with all computations
            client.close()
        logger.info("All task completed. Time taken: %s seconds", time.time() - t_start)


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
        "--nan_limit",
        action="store",
        type=float,
        default=0.005,
        help=("Limit the ratio of NaN (missing data) in covariates. "
              "Only those with NaN rate lower than the limit ratio can be selected during multivariate grid searching."
              "Defaults to 0.5%."
        )
    )
    parser.add_argument(
        "--accelerator", action="store_true", help="Use accelerator automatically"
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="Use early stopping during model fitting",
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
