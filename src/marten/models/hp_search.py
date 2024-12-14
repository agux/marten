import os

OPENBLAS_NUM_THREADS = 1
os.environ["OPENBLAS_NUM_THREADS"] = f"{OPENBLAS_NUM_THREADS}"

import time
import random
import datetime
import argparse
import json
import math
import multiprocessing
import pandas as pd
import numpy as np
from dotenv import load_dotenv

import dask
from dask.distributed import (
    get_worker,
    worker_client,
    as_completed,
    Semaphore,
    wait,
)

from marten.models.base_model import BaseModel
from marten.utils.database import get_database_engine, columns_with_prefix
from marten.utils.logger import get_logger
from marten.utils.worker import (
    await_futures,
    init_client,
    num_workers,
    hps_task_callback,
)
from marten.utils.neuralprophet import select_topk_features
from marten.utils.softs import is_large_model
from marten.utils.trainer import (
    select_device,
    select_randk_covars,
    get_accelerator_locks,
)
from marten.models.worker_func import (
    fit_with_covar,
    log_metrics_for_hyper_params,
    validate_hyperparams,
    get_hpid,
)

from sqlalchemy import text
import psycopg2.extras

from neuralprophet import set_log_level

from sklearn.model_selection import ParameterGrid

from mango import Tuner, scheduler

from scipy.stats import uniform

default_params = {
    # default hyperparameters. the order of keys MATTER (which affects the PK in table)
    "ar_layers": [],
    "batch_size": None,
    "lagged_reg_layers": [],
    "n_lags": 0,
    "yearly_seasonality": "auto",
}

random_seed = 7
logger = None
alchemyEngine = None
args = None
client = None
futures = []
model: BaseModel = None


def init(args):
    global client, model

    _init_local_resource()

    client = init_client(
        __name__,
        int(multiprocessing.cpu_count() * 0.8) if args.worker < 1 else args.worker,
        dashboard_port=args.dashboard_port,
    )

    match args.model.lower():
        case "timemixer":
            from marten.models.time_mixer import TimeMixerModel

            model = TimeMixerModel()
        case "tsmixerx":
            from marten.models.nf_tstimerx import TSMixerxModel

            model = TSMixerxModel()
        case _:
            model = None


def _init_local_resource():
    global logger, alchemyEngine
    # NeuralProphet: Disable logging messages unless there is an error
    set_log_level("ERROR")

    if alchemyEngine is None:
        load_dotenv()  # take environment variables from .env.

        DB_USER = os.getenv("DB_USER")
        DB_PASSWORD = os.getenv("DB_PASSWORD")
        DB_HOST = os.getenv("DB_HOST")
        DB_PORT = os.getenv("DB_PORT")
        DB_NAME = os.getenv("DB_NAME")

        db_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        alchemyEngine = get_database_engine(db_url)

    if logger is None:
        logger = get_logger(__name__)


def load_anchor_ts(symbol, limit, alchemyEngine, cutoff_date=None, anchor_table=None):
    ## support arbitrary types of symbol (could be from different tables, with different features available)
    tbl_cols_dict = {
        "index_daily_em_view": (
            "date DS, change_rate y, vol_change_rate, amt_change_rate, open, close, high, low, volume, amount, "
            "open_preclose_rate, high_preclose_rate, low_preclose_rate"
        ),
        "fund_etf_daily_em_view": (
            "date DS, change_rate y, vol_change_rate, amt_change_rate, open, close, high, low, volume, "
            "turnover, turnover_rate, turnover_change_rate, open_preclose_rate, high_preclose_rate, low_preclose_rate"
        ),
        "us_index_daily_sina_view": (
            "date DS, change_rate y, amt_change_rate amt_cr, open, close, high, low, volume, amount, "
            "vol_change_rate, open_preclose_rate, high_preclose_rate, low_preclose_rate"
        ),
        "cn_bond_indices_view": (
            "date DS, wealthindex_change y, fullpriceindex_change, cleanpriceindex_change, "
            "fullprice, cleanprice, wealth, avgmv_duration, avgcf_duration, avgmv_convexity, avgcf_convexity, "
            "avgcf_ytm, avgmv_ytm, avgbpv, avgmaturity, avgcouponrate, indexprevdaymv, spotsettlementvolume, "
            "avgmv_duration_change_rate, avgcf_duration_change_rate, avgmv_convexity_change_rate, avgcf_convexity_change_rate, "
            "avgcf_ytm_change_rate, avgmv_ytm_change_rate, avgbpv_change_rate, avgmaturity_change_rate, avgcouponrate_change_rate, "
            "indexprevdaymv_change_rate, spotsettlementvolume_change_rate"
        ),
        "hk_index_daily_em_view": (
            "date DS, change_rate y, open, close, high, low, open_preclose_rate, high_preclose_rate, low_preclose_rate"
        ),
        "bond_zh_hs_daily_view": (
            "date DS, change_rate y, open, high, low, close, volume, open_preclose_rate, high_preclose_rate, low_preclose_rate, vol_change_rate"
        ),
        "stock_zh_a_hist_em_view": (
            "date DS, change_rate y, open, close, high, low, volume, turnover, amplitude, change_amt, turnover_rate, turnover_change_rate, "
            "open_preclose_rate, high_preclose_rate, low_preclose_rate, vol_change_rate, amt_change_rate"
        ),
    }
    if anchor_table is None:
        # lookup which table the symbol's data is in
        anchor_table = "index_daily_em_view"  # Default table, replace with actual logic if necessary
        with alchemyEngine.connect() as conn:
            results = conn.execute(
                text("""SELECT "table" FROM symbol_dict WHERE symbol = :symbol"""),
                {"symbol": symbol},
            )
            for row in results:
                table_name = row[0]
                if table_name in tbl_cols_dict:
                    anchor_table = table_name
                    break

    # load anchor TS
    query = f"""
        SELECT {tbl_cols_dict[anchor_table]}
        FROM {anchor_table}
        where symbol = %(symbol)s
    """
    params = {"symbol": symbol}

    if cutoff_date is not None:
        query += " and date <= %(cutoff_date)s"
        params["cutoff_date"] = cutoff_date

    query += " order by DS desc"

    if limit > 0:
        query += " limit %(limit)s"
        params["limit"] = limit

    # re-order the date in ascending order
    query = f"""
        with cte as ({query})
        select * from cte 
        where y is not null
        order by ds
    """

    df = pd.read_sql(
        query,
        alchemyEngine,
        params=params,
        parse_dates=["ds"],
    )

    return df, anchor_table


def covar_symbols_from_table(
    model,
    anchor_symbol,
    dates,
    table,
    feature,
    ts_date,
    min_count,
    sem,
):
    worker = get_worker()
    alchemyEngine, logger = worker.alchemyEngine, worker.logger

    params = {
        "table": table,
        "anchor_symbol": anchor_symbol,
        "feature": feature,
        "ts_date": ts_date,
        "dates": list(dates),
        "min_count": min_count,
    }
    match model:
        case "NeuralProphet":
            existing_cov_symbols = """
                SELECT
                    cov_symbol
                FROM
                    neuralprophet_corel nc
                WHERE
                    AND nc.symbol = %(anchor_symbol)s
                    AND nc.cov_table = %(table)s
                    AND nc.feature = %(feature)s
                    AND nc.ts_date = %(ts_date)s
            """
        case _:
            existing_cov_symbols = """
                SELECT
                    cov_symbol
                FROM
                    paired_correlation pc
                WHERE
                    pc.model = %(model)s
                    AND pc.symbol = %(anchor_symbol)s
                    AND pc.cov_table = %(table)s
                    AND pc.feature = %(feature)s
                    AND pc.ts_date = %(ts_date)s
            """
            params["model"] = model

    # get a list of symbols from the given table, of which metrics are not recorded yet
    if table.startswith("ta_"):
        symbol_col = """t."table" || '::' || t.symbol """
        exclude = ""
        column_names = columns_with_prefix(alchemyEngine, table, feature)
        notnull = (
            "(" + " or ".join([f"t.{c} is not null" for c in column_names]) + ")"
        )
        group_by = 'group by t."table", t.symbol'
    else:
        symbol_col = "t.symbol"
        exclude = "t.symbol <> %(anchor_symbol)s"
        notnull = f"t.{feature} is not null"
        group_by = "group by t.symbol"

    orig_table = (
        table[:-5] if table.startswith("ta_") and table.endswith("_view") else table
    )

    query = f"""
        WITH existing_cov_symbols AS ({existing_cov_symbols})
        select
            {symbol_col} symbol, count(*) num
        from
            {orig_table} t
        where
            NOT EXISTS (
                SELECT 1
                FROM existing_cov_symbols ecs
                WHERE {symbol_col} = ecs.cov_symbol
            )
            and {notnull}
            and {exclude}
            and t.date = ANY(%(dates)s::date[])
        {group_by}
        having 
            count(*) >= %(min_count)s
    """

    if sem and table.startswith("ta_"):
        with sem:
            raw_conn = alchemyEngine.raw_connection()
            try:
                cursor = raw_conn.cursor()
                final_query = cursor.mogrify(query, params)
                logger.info(final_query)
                cursor.close()
            finally:
                raw_conn.close()
                
            with alchemyEngine.connect() as conn:
                cov_symbols = pd.read_sql(query, conn, params=params)
    else:
        with alchemyEngine.connect() as conn:
            cov_symbols = pd.read_sql(query, conn, params=params)

    cov_symbols = cov_symbols[["symbol"]]
    cov_symbols.drop_duplicates(subset=["symbol"], inplace=True)
    cov_symbols = cov_symbols["symbol"].tolist()

    logger.info(
        "Identified %s candidate covariate symbols for feature %s.%s",
        len(cov_symbols),
        table,
        feature,
    )

    return cov_symbols, feature


def _pair_endogenous_covar_metrics(
    anchor_symbol, anchor_df, cov_table, features, args, cutoff_date, sem, locks
):
    global client, futures, logger, alchemyEngine

    # remove feature elements already exists in the neuralprophet_corel table.
    features = _remove_measured_features(
        alchemyEngine, args.model, anchor_symbol, cov_table, features, cutoff_date
    )

    if not features:
        return

    for feature in features:
        logger.debug(
            "submitting fit_with_covar:\nanchor_symbol:%s\ncov_table:%s\ncov_symbol:%s\nfeature:%s",
            anchor_symbol,
            cov_table,
            anchor_symbol,
            feature,
        )
        future = client.submit(
            fit_with_covar,
            anchor_symbol,
            anchor_df,
            cov_table,
            anchor_symbol,
            anchor_df["ds"].min().strftime("%Y-%m-%d"),
            args.random_seed,
            feature,
            select_device(
                args.accelerator,
                getattr(args, "gpu_util_threshold", None),
                getattr(args, "gpu_ram_threshold", None),
            ),
            args.early_stopping,
            args.infer_holiday,
            sem,
            locks,
        )
        futures.append(future)
        await_futures(futures, False)


def _pair_covar_metrics(
    client,
    anchor_symbol,
    anchor_df_future,
    cov_table,
    cov_symbols,
    feature,
    min_date,
    args,
    sem,
    locks,
):
    covar_fut = []
    worker = get_worker()
    logger = worker.logger
    for symbol in cov_symbols:
        logger.debug(
            "submitting fit_with_covar: %s @ %s.%s",
            symbol,
            cov_table,
            feature,
        )
        covar_fut.append(
            client.submit(
                fit_with_covar,
                anchor_symbol,
                anchor_df_future,
                cov_table,
                symbol,
                min_date,
                args.random_seed,
                feature,
                select_device(
                    args.accelerator,
                    getattr(args, "gpu_util_threshold", None),
                    getattr(args, "gpu_ram_threshold", None),
                ),
                args.early_stopping,
                args.infer_holiday,
                sem=sem,
                locks=locks,
                key=f"{fit_with_covar.__name__}-{symbol}@{cov_table}.{feature}",
                priority=10,
            )
        )
        # if too much pending task, then slow down for the tasks to be digested
        # await_futures(covar_fut, False)
    wait(covar_fut)
    # await_futures(covar_fut, hard_wait=True)


def _load_covar_set(covar_set_id, model, alchemyEngine):
    params = {
        "covar_set_id": covar_set_id,
    }
    match model:
        case "NeuralProphet":
            join_clause = """
                JOIN
                    neuralprophet_corel t2
                ON
                    t1.symbol = t2.symbol
                    AND t1.cov_table = t2.cov_table
                    AND t1.cov_symbol = t2.cov_symbol
                    AND t1.cov_feature = t2.feature
            """
        case _:
            join_clause = """
                JOIN
                    paired_correlation t2
                ON
                    t1.symbol = t2.symbol
                    AND t1.cov_table = t2.cov_table
                    AND t1.cov_symbol = t2.cov_symbol
                    AND t1.cov_feature = t2.feature
                    AND t2.model = %(model)s
            """
            params["model"] = model
    query = f"""
        WITH ranked_data AS (
            SELECT
                t1.cov_symbol,
                t1.cov_table,
                t1.cov_feature feature,
                t2.ts_date,
                t2.loss_val,
                t2.nan_count,
                ROW_NUMBER() OVER (
                    PARTITION BY t1.symbol, t1.cov_table, t1.cov_symbol, t1.cov_feature
                    ORDER BY t2.ts_date DESC, t2.loss_val ASC, t2.nan_count ASC
                ) AS rnk
            FROM
                covar_set t1
            {join_clause}
            WHERE
                t1.id = %(covar_set_id)s
        )
        SELECT
            cov_symbol,
            cov_table,
            feature
        FROM
            ranked_data
        WHERE
            rnk = 1
        ORDER BY
            loss_val ASC, nan_count ASC, cov_table, cov_symbol
    """
    df = pd.read_sql(
        query,
        alchemyEngine,
        params=params,
    )
    return df


def _load_covars(
    alchemyEngine,
    max_covars,
    anchor_symbol,
    nan_threshold=None,
    cov_table=None,
    feature=None,
    cutoff_date=None,
    model=None,
):
    global logger
    params = {
        "anchor_symbol": anchor_symbol,
        "ts_date": cutoff_date,
    }
    match model:
        case "NeuralProphet":
            sub_query = """
                select
                    loss_val
                from
                    neuralprophet_corel
                where
                    symbol = %(anchor_symbol)s
                    and cov_symbol = %(anchor_symbol)s
                    and feature = 'y'
                    and ts_date <= %(ts_date)s
                    order by ts_date desc
                    limit 1
            """
            query = f"""
                select DISTINCT ON (ts_date, loss_val, nan_count, cov_table, cov_symbol)
                    cov_symbol, cov_table, feature
                from
                    neuralprophet_corel
                where
                    symbol = %(anchor_symbol)s
                    and loss_val < ({sub_query})
            """
        case _:
            sub_query = """
                select
                    loss_val
                from
                    paired_correlation
                where
                    model = %(model)s
                    and symbol = %(anchor_symbol)s
                    and cov_symbol = %(anchor_symbol)s
                    and feature = 'y'
                    and ts_date <= %(ts_date)s
                    order by ts_date desc
                    limit 1
            """
            query = f"""
                select DISTINCT ON (ts_date, loss_val, nan_count, cov_table, cov_symbol)
                    cov_symbol, cov_table, feature
                from
                    paired_correlation
                where
                    model = %(model)s
                    and symbol = %(anchor_symbol)s
                    and loss_val < ({sub_query})
            """
            params["model"] = model
    limit_clause = ""
    if max_covars > 0:
        params["limit"] = max_covars
        limit_clause = " limit %(limit)s"
    if cov_table is not None:
        query += " and cov_table = %(cov_table)s"
        params["cov_table"] = cov_table
    if feature is not None:
        query += " and feature = %(feature)s"
        params["feature"] = feature
    if nan_threshold is not None:
        query += " and nan_count < %(nan_threshold)s"
        params["nan_threshold"] = nan_threshold

    # Note: SELECT DISTINCT ON expressions must match initial ORDER BY expressions
    query += " ORDER BY ts_date desc, loss_val, nan_count, cov_table, cov_symbol"
    query += limit_clause
    logger.debug("loading topk covariates using sql:\n%s\nparams:%s", query, params)
    df = pd.read_sql(
        query,
        alchemyEngine,
        params=params,
    )

    if df.empty:
        logger.warning(
            "no covariates found for %s\nSQL: %s\nParameters:%s",
            anchor_symbol,
            query,
            params,
        )
        return df, -1

    with (
        alchemyEngine.begin() as conn
    ):  # need to use transaction for inserting new covar_set records.
        # check if the same set of covar features exists in `covar_set` table. If so, reuse the same set_id.
        query = """
            select id, count(*) num
            from covar_set
            where 
                symbol = :symbol
                and id in (
                    select id 
                    from covar_set
                    where
                        symbol = :symbol
                        and (cov_symbol, cov_table, cov_feature) IN :values
                )
            group by id
            having count(*) = :num
            order by id desc
        """
        params = {
            "symbol": anchor_symbol,
            "values": tuple(df.itertuples(index=False, name=None)),
            "num": len(df),
        }
        result = conn.execute(text(query), params)
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


def _load_covar_feature(cov_table, feature, symbols):
    worker = get_worker()
    alchemyEngine = worker.alchemyEngine
    match cov_table:
        case "bond_metrics_em" | "bond_metrics_em_view":
            query = f"""
                SELECT 'bond' ID, date DS, {feature} y
                FROM {cov_table}
                order by DS asc
            """
            table_feature_df = pd.read_sql(query, alchemyEngine, parse_dates=["ds"])
        case "currency_boc_safe_view":
            query = f"""
                SELECT 'currency_exchange' ID, date DS, {feature} y
                FROM {cov_table}
                order by DS asc
            """
            table_feature_df = pd.read_sql(query, alchemyEngine, parse_dates=["ds"])
        case _ if cov_table.startswith("ta_"):  # handle technical indicators table
            column_names = columns_with_prefix(alchemyEngine, cov_table, feature)
            # query rows from the TA table
            query = f"""
                SELECT symbol ID, date DS, {', '.join(column_names)}
                FROM {cov_table}
                where symbol in %(symbols)s
                order by ID, DS asc
            """
            params = {
                "symbols": tuple(symbols),
            }
            table_feature_df = pd.read_sql(
                query, alchemyEngine, params=params, parse_dates=["ds"]
            )
        case _:
            query = f"""
                SELECT symbol ID, date DS, {feature} y
                FROM {cov_table}
                where symbol in %(symbols)s
                order by ID, DS asc
            """
            params = {
                "symbols": tuple(symbols),
            }
            table_feature_df = pd.read_sql(
                query, alchemyEngine, params=params, parse_dates=["ds"]
            )
    return table_feature_df


def augment_anchor_df_with_covars(df, args, alchemyEngine, logger, cutoff_date):
    global client
    # date_col = "ds" if args.model == "NeuralProphet" else "date"
    merged_df = df[["ds", "y"]]
    if args.covar_set_id is not None:
        covar_set_id = args.covar_set_id
        covars_df = _load_covar_set(covar_set_id, args.model, alchemyEngine)
    else:
        nan_threshold = round(len(df) * args.nan_limit, 0)
        logger.info(
            "covar_set_id is not provided, selecting top %s covars with NaN threshold %s, cutoff date %s",
            args.max_covars,
            nan_threshold,
            cutoff_date,
        )
        covars_df, covar_set_id = _load_covars(
            alchemyEngine,
            args.max_covars,
            args.symbol,
            nan_threshold,
            cutoff_date=cutoff_date,
            model=args.model,
        )

    if covars_df.empty:
        table = (
            "neuralprophet_corel"
            if args.model.lower() == "neuralprophet"
            else "paired_correlation"
        )
        raise Exception(
            f"No qualified covariates can be found for {args.symbol}. Please check the data in table {table}"
        )

    logger.info(
        "loaded top %s qualified covariates. covar_set id: %s",
        len(covars_df),
        covar_set_id,
    )

    ranked_features = [
        f"{r["feature"]}::{r["cov_table"]}::{r["cov_symbol"]}"
        for _, r in covars_df.iterrows()
    ]

    # covars_df contain these columns: cov_symbol, cov_table, feature
    by_table_feature = covars_df.groupby(["cov_table", "feature"])
    futures = []
    for group1, sdf1 in by_table_feature:
        ## load covariate time series from different tables and/or features
        cov_table = group1[0]
        feature = group1[1]
        futures.append(
            client.submit(_load_covar_feature, cov_table, feature, sdf1["cov_symbol"])
        )

    table_feature_dfs = client.gather(futures)

    for (group1, sdf1), table_feature_df in zip(by_table_feature, table_feature_dfs):
        ## load covariate time series from different tables and/or features
        cov_table = group1[0]
        feature = group1[1]
        # table_feature_df = _load_covar_feature(cov_table, feature, sdf1["cov_symbol"])

        # merge and append the feature column of table_feature_df to merged_df, by matching dates
        # split table_feature_df by symbol column
        grouped = table_feature_df.groupby("id")
        for group2, sdf2 in grouped:
            if "y" in sdf2.columns:
                col_name = f"{feature}::{cov_table}::{group2}"
                sdf2.rename(
                    columns={
                        "y": col_name,
                    },
                    inplace=True,
                )
                sdf2 = sdf2[["ds", col_name]]
            else:
                col_names = {}
                for col in [c for c in df.columns if c.startswith(f"{feature}_")]:
                    col_names[col] = f"{col}::{cov_table}::{group2}"
                sdf2.rename(columns=col_names, inplace=True)
                sdf2 = sdf2[["ds"] + list(col_names.values())]
            merged_df = pd.merge(merged_df, sdf2, on="ds", how="left")

    missing_values = merged_df.isna().sum()
    missing_values = missing_values[missing_values > 0]
    logger.info("Count of missing values:\n%s", missing_values)

    return merged_df, covar_set_id, ranked_features


def _get_layers(w_power=6, min_size=2, max_size=20):
    layers = []
    # Loop over powers of 2 from 2^1 to 2^6
    for i in range(1, w_power + 1):
        power_of_two = 2**i
        # Loop over list lengths from 2 to 20
        for j in range(min_size, max_size + 1):
            # Create a list with the current power of two, repeated 'j' times
            element = [power_of_two] * j
            # Append the list to the result
            layers.append(element)
    return layers


def _search_space(max_covars):
    # ar_layers, lagged_reg_layers = _get_layers(10, 1, 64), _get_layers(10, 1, 64)

    # ss = dict(
    #     batch_size=[None, 100, 200, 300, 400, 500],
    #     n_lags=list(range(0, 60+1)),
    #     yearly_seasonality=["auto"] + list(range(1, 60+1)),
    #     # ar_layers=[[]] + ar_layers,
    #     # lagged_reg_layers=[[]] + lagged_reg_layers,
    #     ar_layer_spec=[None] + [[2**w, d] for w in range(1, 10+1) for d in range(1, 64+1)],
    #     lagged_reg_layer_spec=[None] + [[2**w, d] for w in range(1, 10+1) for d in range(1, 64+1)],
    #     topk_covar=list(range(2, max_covars+1)),
    #     optimizer=["AdamW", "SGD"],
    # )

    # NOTE buggy HP in neuralprophet:
    # trend_reg_threshold=[True, False]

    ss = f"""dict(
        growth=["linear", "discontinuous"],
        batch_size=[None, 100, 200, 300, 400, 500],
        n_lags=list(range(0, 60+1)),
        yearly_seasonality=["auto"] + list(range(1, 60+1)),
        ar_layer_spec=[None] + [[2**w, d] for w in range(1, 10+1) for d in range(1, 64+1)],
        ar_reg=uniform(0, 100),
        lagged_reg_layer_spec=[None] + [[2**w, d] for w in range(1, 10+1) for d in range(1, 64+1)],
        topk_covar=list(range(0, {max_covars}+1)),
        optimizer=["AdamW", "SGD"],
        trend_reg=uniform(0, 100),
        trend_reg_threshold=[True, False],
        seasonality_reg=uniform(0, 100),
        seasonality_mode=["additive", "multiplicative"],
        normalize=["off", "standardize", "soft", "soft1"],
    )"""

    return ss


def _init_search_grid():
    global logger

    ar_layers, lagged_reg_layers = _get_layers(), _get_layers()

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
            "batch_size": [100, 200, 300, 400, 500],
            "n_lags": list(range(1, 31)),
            "yearly_seasonality": list(range(3, 31)),
            "ar_layers": ar_layers,
            "lagged_reg_layers": lagged_reg_layers,
        },
    ]
    grid = ParameterGrid(param_grid)
    logger.info("size of grid: %d", len(grid))
    return grid, param_grid


def _cleanup_stale_keys():
    global alchemyEngine, logger
    with alchemyEngine.begin() as conn:
        conn.execute(
            text(
                """
                delete from hps_metrics
                where loss_val is null 
                    and last_modified <= NOW() - INTERVAL '1 hour'
                """
            )
        )


def hp_deserializer(dct):
    tuple_props = ["ar_layer_spec", "lagged_reg_layer_spec"]

    for key, value in dct.items():
        if key in tuple_props and isinstance(value, list):
            dct[key] = tuple(value)
    return dct


def preload_warmstart_tuples(
    model_name, anchor_symbol, covar_set_id, hps_id, limit, feat_size
):
    global alchemyEngine, logger, model

    with alchemyEngine.connect() as conn:
        results = conn.execute(
            text(
                """
                select hyper_params, loss_val, sub_topk
                from hps_metrics
                where model = :model
                    and anchor_symbol = :anchor_symbol
                    and covar_set_id = :covar_set_id
                    and hps_id = :hps_id
                    and loss_val is not null
                order by loss_val
            """
                # limit :limit
            ),
            {
                "model": model_name,
                "anchor_symbol": anchor_symbol,
                "covar_set_id": covar_set_id,
                "hps_id": hps_id,
                # "limit": limit,
            },
        )

        tuples = []
        for row in results:
            # param_dict = json.loads(row[0], object_hook=hp_deserializer)
            param_dict = row[0]
            match model_name:
                case "NeuralProphet":
                    # Fill in default values if not exists in historical HP
                    # To match the size of tensors in bayesopt
                    if "optimizer" not in param_dict:
                        param_dict["optimizer"] = "AdamW"
                    if "growth" not in param_dict:
                        param_dict["growth"] = "linear"
                    if "ar_reg" not in param_dict:
                        param_dict["ar_reg"] = 0
                    if "trend_reg" not in param_dict:
                        param_dict["trend_reg"] = 0
                    if "trend_reg_threshold" not in param_dict:
                        param_dict["trend_reg_threshold"] = False
                    if "seasonality_reg" not in param_dict:
                        param_dict["seasonality_reg"] = 0
                    if "seasonality_mode" not in param_dict:
                        param_dict["seasonality_mode"] = "additive"
                    if "normalize" not in param_dict:
                        param_dict["normalize"] = "soft"
                case "SOFTS":
                    # param_dict["covar_dist"] = 0.0
                    if "covar_dist" not in param_dict:
                        # TODO can we use fabricated list instead of real dirichlet sample?
                        param_dict["covar_dist"] = np.full(
                            feat_size, 1.0 / float(feat_size)
                        )
                    else:
                        param_dict["covar_dist"] = np.array(param_dict["covar_dist"])

                    # logger.info("""param_dict: %s""", param_dict)
                case _:
                    param_dict = model.restore_params(
                        params=param_dict, feat_size=feat_size
                    )

            tuples.append((param_dict, row[1]))

        return tuples if len(tuples) > 0 else None


def power_demand(args, params):
    global model
    match args.model:
        case "NeuralProphet":
            return 1
        case "SOFTS":
            return 2 if is_large_model(params, params["topk_covar"]) else 1
        case _:
            return model.power_demand(args, params)


def _bayesopt_run(
    df,
    n_jobs,
    covar_set_id,
    hps_id,
    ranked_features,
    space,
    args,
    iteration,
    domain_size,
    resume,
    locks,
):
    global logger, client

    @scheduler.custom(n_jobs=n_jobs)
    def objective(params_batch):
        jobs = []
        t1 = time.time()
        nworker = num_workers(False)
        client.set_metadata(["workload_info", "total"], len(params_batch))
        client.set_metadata(["workload_info", "workers"], nworker)
        client.set_metadata(["workload_info", "finished"], 0)
        for i, params in enumerate(params_batch):
            new_df = df
            hpid, _ = get_hpid(params)
            if "topk_covar" in params:
                if "covar_dist" in params:
                    new_df = select_randk_covars(
                        df, ranked_features, params["covar_dist"], params["topk_covar"]
                    )
                else:
                    new_df = select_topk_features(
                        df, ranked_features, params["topk_covar"]
                    )
            future = client.submit(
                validate_hyperparams,
                args,
                new_df,
                covar_set_id,
                hps_id,
                params,
                resources={"POWER": power_demand(args, params)},
                retries=1,
                locks=locks,
                key=f"{validate_hyperparams.__name__}-{hpid}",
            )
            future.add_done_callback(hps_task_callback)
            jobs.append(future)
            if i < nworker:
                interval = random.randint(5000, 15000) / 1000.0
                time.sleep(interval)
        results = client.gather(jobs, errors="skip")
        elapsed = round(time.time() - t1, 3)
        # logger.info("gathered results type %s, len: %s", type(results), len(results))
        # logger.info("gathered results: %s", results)
        results = [(p, l) for p, l in results if p is not None and l is not None]
        if len(results) == 0:
            logger.warning(
                "Results not available at this iteration. Elapsed: %s", elapsed
            )
            return [], []
        params, loss = zip(*results)
        params = list(params)
        loss = list(loss)
        logger.info("Elapsed: %s, Successful results: %s", elapsed, len(results))
        # restart client here to free up memory
        if args.restart_workers:
            client.restart()
        return params, loss

    warmstart_tuples = None
    if resume:
        warmstart_tuples = preload_warmstart_tuples(
            args.model,
            args.symbol,
            covar_set_id,
            hps_id,
            args.batch_size * iteration * 2,
            len(ranked_features),
        )
    if warmstart_tuples is not None:
        logger.info(
            "preloaded %s historical searched hyper-params for warm-start.",
            len(warmstart_tuples),
        )
    else:
        logger.info("no available historical data for warm-start")
    tuner = Tuner(
        space,
        objective,
        dict(
            initial_random=n_jobs,
            batch_size=n_jobs,
            num_iteration=iteration,
            initial_custom=warmstart_tuples,
            domain_size=domain_size,
        ),
    )
    results = tuner.minimize()
    logger.info("best parameters: %s", results["best_params"])
    logger.info("best objective: %s", results["best_objective"])
    return results["best_objective"]


def bayesopt(df, covar_set_id, hps_id, ranked_features):
    global logger, args

    _cleanup_stale_keys()

    space_str = _search_space(args.max_covars)

    # Convert args to a dictionary, excluding non-serializable items
    # FIXME: no need to update the table each time, especially in resume mode?
    args_dict = {k: v for k, v in vars(args).items() if not callable(v)}
    # space_json = json.dumps(space, sort_keys=True)
    args_json = json.dumps(args_dict, sort_keys=True)
    update_hps_sessions(hps_id, "bayesopt", args_json, space_str, covar_set_id)

    n_jobs = args.batch_size

    # split large iterations into smaller runs to avoid OOM / memory leak
    mango_itr = min(2, int(1000.0 / args.batch_size))
    for i in range(0, math.ceil(args.iteration / mango_itr)):
        itr = min(mango_itr, args.iteration - i * mango_itr)
        _bayesopt_run(
            df,
            n_jobs,
            covar_set_id,
            hps_id,
            ranked_features,
            eval(space_str, {"uniform": uniform}),
            args,
            itr,
            i > 0 or args.resume,
        )


def grid_search(df, covar_set_id, hps_id, ranked_features):
    global alchemyEngine, logger, random_seed, client, futures, args

    _cleanup_stale_keys()

    grid, raw_list = _init_search_grid()

    space_json = json.dumps(raw_list, sort_keys=True)
    args_json = json.dumps(vars(args), sort_keys=True)
    update_hps_sessions(hps_id, "grid_search", args_json, space_json)

    for params in grid:
        future = client.submit(
            log_metrics_for_hyper_params,
            args.symbol,
            df,
            params,
            args.epochs,
            random_seed,
            select_device(
                args.accelerator,
                getattr(args, "gpu_util_threshold", None),
                getattr(args, "gpu_ram_threshold", None),
            ),
            covar_set_id,
            hps_id,
            args.early_stopping,
            args.infer_holiday,
        )
        futures.append(future)
        # if too much pending task, then slow down for the tasks to be digested.
        await_futures(futures, False)

    await_futures(futures)
    # All tasks have completed at this point


def update_hps_sessions(id, method, search_params, search_space, covar_set_id):
    global alchemyEngine
    update = """
        update hps_sessions
        set
            "method" = :method,
            search_params = :search_params,
            search_space = :search_space,
            covar_set_id = :covar_set_id
        where
            id = :id
    """
    params = {
        "method": method,
        "search_params": search_params,
        "search_space": search_space,
        "covar_set_id": covar_set_id,
        "id": id,
    }
    with alchemyEngine.begin() as conn:
        conn.execute(text(update), params)


def _remove_measured_features(alchemyEngine, model, anchor_symbol, cov_table, features, ts_date=None):
    params = {
        "symbol": anchor_symbol,
        "cov_table": cov_table,
        "features": tuple(features),
    }
    match model:
        case "NeuralProphet":
            query = """
                select feature
                from neuralprophet_corel
                where symbol = %(symbol)s
                and cov_table = %(cov_table)s
                and feature in %(features)s
            """
        case _:
            query = """
                select feature
                from paired_correlation
                where 
                    model = %(model)s
                    and symbol = %(symbol)s
                    and cov_table = %(cov_table)s
                    and feature in %(features)s
            """
            params["model"] = model
    if ts_date is not None:
        query += " and ts_date = %(ts_date)s"
        params["ts_date"] = ts_date
    with alchemyEngine.connect() as conn:
        existing_features_pd = pd.read_sql(
            query,
            conn,
            params=params,
        )
    # remove elements in the `features` list that exist in the existing_features_pd.
    features = list(set(features) - set(existing_features_pd["feature"]))
    return features


def covar_metric(
    anchor_symbol, anchor_df, cov_table, features, dates, min_count, args, sem, locks
):
    worker = get_worker()
    logger = worker.logger
    alchemyEngine = worker.alchemyEngine
    min_date = min(dates).strftime("%Y-%m-%d")
    cutoff_date = max(dates).strftime("%Y-%m-%d")

    if cov_table.startswith("bond_metrics_em") or cov_table.startswith(
        "currency_boc_safe"
    ):
        features = _remove_measured_features(
            alchemyEngine, args.model, anchor_symbol, cov_table, features, cutoff_date
        )

    if len(features) == 0:
        logger.info(
            "no new features in %s need to be calculated for %s on cutoff date %s",
            cov_table,
            anchor_symbol,
            cutoff_date,
        )
        return None

    logger.info(
        "looking for covariate symbols for %s features in %s", len(features), cov_table
    )

    cov_symbols_fut = []
    num_symbols = 0
    with worker_client() as client:
        for feature in features:
            match cov_table:
                case "bond_metrics_em" | "bond_metrics_em_view":
                    # construct a dummy cov_symbols dataframe with `symbol` column and the value 'bond'.
                    _pair_covar_metrics(
                        client,
                        anchor_symbol,
                        anchor_df,
                        cov_table,
                        ["bond"],
                        feature,
                        min_date,
                        args,
                        sem,
                        locks,
                    )
                    num_symbols += 1
                case "currency_boc_safe_view":
                    _pair_covar_metrics(
                        client,
                        anchor_symbol,
                        anchor_df,
                        cov_table,
                        ["currency_exchange"],
                        feature,
                        min_date,
                        args,
                        sem,
                        locks,
                    )
                    num_symbols += 1
                case _:
                    cov_symbols_fut.append(
                        client.submit(
                            covar_symbols_from_table,
                            args.model,
                            anchor_symbol,
                            dates,
                            cov_table,
                            feature,
                            cutoff_date,
                            min_count,
                            sem,
                            key=f"{covar_symbols_from_table.__name__}-{cov_table}.{feature}",
                            priority=2,
                        )
                    )
                    # cov_symbols = _covar_symbols_from_table(
                    #     args.model,
                    #     anchor_symbol,
                    #     dates,
                    #     cov_table,
                    #     feature,
                    #     cutoff_date,
                    #     min_count,
                    # )
                    # remove duplicate records in cov_symbols dataframe, by checking the `symbol` column values.
                    # cov_symbols.drop_duplicates(subset=["symbol"], inplace=True)
        # logger.info("[DEBUG] len(futures): %s in %s", len(cov_symbols_fut), cov_table)

        if cov_symbols_fut:
            for batch in as_completed(cov_symbols_fut).batches():
                for future in batch:
                    cov_symbols, feature = future.result()
                    # logger.info(
                    #     "identified %s symbols for %s.%s",
                    #     len(cov_symbols),
                    #     cov_table,
                    #     feature
                    # )
                    _pair_covar_metrics(
                        client,
                        anchor_symbol,
                        anchor_df,
                        cov_table,
                        cov_symbols,
                        feature,
                        min_date,
                        args,
                        sem,
                        locks,
                    )
                    num_symbols += len(cov_symbols)
    logger.info(
        "finished covar_metric for %s features in %s, total covar symbols: %s",
        len(features),
        cov_table,
        num_symbols,
    )
    return True


def prep_covar_baseline_metrics(anchor_df, anchor_table, args):
    global random_seed, client, futures

    anchor_symbol = args.symbol

    # min_date = anchor_df["ds"].min().strftime("%Y-%m-%d")
    cutoff_date = anchor_df["ds"].max().strftime("%Y-%m-%d")
    min_count = int(len(anchor_df) * (1 - args.nan_limit))
    dates = tuple(anchor_df["ds"])

    dask.config.set({"distributed.scheduler.locks.lease-timeout": "300s"})
    sem = Semaphore(
        max_leases=int(os.getenv("RESOURCE_INTENSIVE_SQL_SEMAPHORE", args.min_worker)),
        name="RESOURCE_INTENSIVE_SQL_SEMAPHORE",
    )
    locks = get_accelerator_locks(0, 1, "60s")

    # endogenous features of the anchor time series per se
    endogenous_features = [col for col in anchor_df.columns if col not in ("ds")]
    _pair_endogenous_covar_metrics(
        anchor_symbol,
        anchor_df,
        anchor_table,
        endogenous_features,
        args,
        cutoff_date,
        sem,
        locks,
    )

    # for the rest of exogenous covariates, keep only the core features of anchor_df
    anchor_df = anchor_df[["ds", "y"]]

    # prep CN index covariates
    features = [
        "change_rate",
        "amt_change_rate",
        "vol_change_rate",
        "open_preclose_rate",
        "high_preclose_rate",
        "low_preclose_rate",
    ]
    cov_table = "index_daily_em_view"
    futures.append(
        client.submit(
            covar_metric,
            anchor_symbol,
            anchor_df,
            cov_table,
            features,
            dates,
            min_count,
            args,
            sem,
            locks,
            key=f"{covar_metric.__name__}-{cov_table}({len(features)})",
        )
    )

    # prep ETF covariates fund_etf_daily_em_view
    features = [
        "change_rate",
        "amt_change_rate",
        "vol_change_rate",
        "turnover_rate",
        "turnover_change_rate",
        "open_preclose_rate",
        "high_preclose_rate",
        "low_preclose_rate",
    ]
    cov_table = "fund_etf_daily_em_view"
    futures.append(
        client.submit(
            covar_metric,
            anchor_symbol,
            anchor_df,
            cov_table,
            features,
            dates,
            min_count,
            args,
            sem,
            locks,
            key=f"{covar_metric.__name__}-{cov_table}({len(features)})",
        )
    )

    # prep bond covariates bond_metrics_em, cn_bond_indices_view
    features = [
        "china_yield_2y",
        "china_yield_10y",
        "china_yield_30y",
        "china_yield_spread_10y_2y",
        "us_yield_2y",
        "us_yield_10y",
        "us_yield_30y",
        "us_yield_spread_10y_2y",
        "quantile",
        "china_yield_2y_change_rate",
        "china_yield_5y_change_rate",
        "china_yield_10y_change_rate",
        "china_yield_30y_change_rate",
        "china_yield_spread_10y_2y_change_rate",
        "us_yield_2y_change_rate",
        "us_yield_5y_change_rate",
        "us_yield_10y_change_rate",
        "us_yield_30y_change_rate",
        "us_yield_spread_10y_2y_change_rate",
        "performance_benchmark_change_rate",
    ]
    cov_table = "bond_metrics_em_view"
    futures.append(
        client.submit(
            covar_metric,
            anchor_symbol,
            anchor_df,
            cov_table,
            features,
            dates,
            min_count,
            args,
            sem,
            locks,
            key=f"{covar_metric.__name__}-{cov_table}({len(features)})",
        )
    )

    features = [
        "wealthindex_change",
        "fullpriceindex_change",
        "cleanpriceindex_change",
        "avgmv_duration_change_rate",
        "avgcf_duration_change_rate",
        "avgmv_convexity_change_rate",
        "avgcf_convexity_change_rate",
        "avgcf_ytm_change_rate",
        "avgmv_ytm_change_rate",
        "avgbpv_change_rate",
        "avgmaturity_change_rate",
        "avgcouponrate_change_rate",
        "indexprevdaymv_change_rate",
        "spotsettlementvolume_change_rate",
    ]
    cov_table = "cn_bond_indices_view"
    futures.append(
        client.submit(
            covar_metric,
            anchor_symbol,
            anchor_df,
            cov_table,
            features,
            dates,
            min_count,
            args,
            sem,
            locks,
            key=f"{covar_metric.__name__}-{cov_table}({len(features)})",
        )
    )

    # prep US index covariates us_index_daily_sina
    features = [
        "change_rate",
        "amt_change_rate",
        "vol_change_rate",
        "open_preclose_rate",
        "high_preclose_rate",
        "low_preclose_rate",
    ]
    cov_table = "us_index_daily_sina_view"
    futures.append(
        client.submit(
            covar_metric,
            anchor_symbol,
            anchor_df,
            cov_table,
            features,
            dates,
            min_count,
            args,
            sem,
            locks,
            key=f"{covar_metric.__name__}-{cov_table}({len(features)})",
        )
    )

    # prep HK index covariates hk_index_daily_sina
    features = [
        "change_rate",
        "open_preclose_rate",
        "high_preclose_rate",
        "low_preclose_rate",
    ]
    cov_table = "hk_index_daily_em_view"
    futures.append(
        client.submit(
            covar_metric,
            anchor_symbol,
            anchor_df,
            cov_table,
            features,
            dates,
            min_count,
            args,
            sem,
            locks,
            key=f"{covar_metric.__name__}-{cov_table}({len(features)})",
        )
    )

    # prep CN bond: bond_zh_hs_daily_view
    features = [
        "change_rate",
        "vol_change_rate",
        "open_preclose_rate",
        "high_preclose_rate",
        "low_preclose_rate",
    ]
    cov_table = "bond_zh_hs_daily_view"
    futures.append(
        client.submit(
            covar_metric,
            anchor_symbol,
            anchor_df,
            cov_table,
            features,
            dates,
            min_count,
            args,
            sem,
            locks,
            key=f"{covar_metric.__name__}-{cov_table}({len(features)})",
        )
    )

    # prep CN stock features. Sync table & view column names
    features = [
        "change_rate",
        "turnover_rate",
        "turnover_change_rate",
        "open_preclose_rate",
        "high_preclose_rate",
        "low_preclose_rate",
        "vol_change_rate",
        "amt_change_rate",
    ]
    cov_table = "stock_zh_a_hist_em_view"
    futures.append(
        client.submit(
            covar_metric,
            anchor_symbol,
            anchor_df,
            cov_table,
            features,
            dates,
            min_count,
            args,
            sem,
            locks,
            key=f"{covar_metric.__name__}-{cov_table}({len(features)})",
        )
    )

    # RMB exchange rate
    features = [
        "usd_change_rate",
        "eur_change_rate",
        "jpy_change_rate",
        "hkd_change_rate",
        "gbp_change_rate",
        "aud_change_rate",
        "nzd_change_rate",
        "sgd_change_rate",
        "chf_change_rate",
        "cad_change_rate",
        "myr_change_rate",
        "rub_change_rate",
        "zar_change_rate",
        "krw_change_rate",
        "aed_change_rate",
        "qar_change_rate",
        "huf_change_rate",
        "pln_change_rate",
        "dkk_change_rate",
        "sek_change_rate",
        "nok_change_rate",
        "try_change_rate",
        "php_change_rate",
        "thb_change_rate",
        "mop_change_rate",
    ]
    cov_table = "currency_boc_safe_view"
    futures.append(
        client.submit(
            covar_metric,
            anchor_symbol,
            anchor_df,
            cov_table,
            features,
            dates,
            min_count,
            args,
            sem,
            locks,
            key=f"{covar_metric.__name__}-{cov_table}({len(features)})",
        )
    )

    # SGE spot
    features = [
        "change_rate",
        "open_preclose_rate",
        "high_preclose_rate",
        "low_preclose_rate",
    ]
    cov_table = "spot_hist_sge_view"
    futures.append(
        client.submit(
            covar_metric,
            anchor_symbol,
            anchor_df,
            cov_table,
            features,
            dates,
            min_count,
            args,
            sem,
            locks,
            key=f"{covar_metric.__name__}-{cov_table}({len(features)})",
        )
    )

    # Interbank interest rates
    features = ["change_rate"]
    cov_table = "interbank_rate_hist_view"
    futures.append(
        client.submit(
            covar_metric,
            anchor_symbol,
            anchor_df,
            cov_table,
            features,
            dates,
            min_count,
            args,
            sem,
            locks,
            key=f"{covar_metric.__name__}-{cov_table}({len(features)})",
        )
    )

    # Technical indicators
    features = [
        "alma",
        "dema",
        "epma",
        "ema",
        "ht_trendline",
        "hma",
        "kama",
        "mama",
        "dynamic",
        "smma",
        "sma",
        "t3",
        "tema",
        "vwma",
        "wma",
        "vwap",
    ]
    cov_table = "ta_ma_view"
    futures.append(
        client.submit(
            covar_metric,
            anchor_symbol,
            anchor_df,
            cov_table,
            features,
            dates,
            min_count,
            args,
            sem,
            locks,
            key=f"{covar_metric.__name__}-{cov_table}({len(features)})",
        )
    )
    features = [
        "slope",
    ]
    cov_table = "ta_numerical_analysis_view"
    futures.append(
        client.submit(
            covar_metric,
            anchor_symbol,
            anchor_df,
            cov_table,
            features,
            dates,
            min_count,
            args,
            sem,
            locks,
            key=f"{covar_metric.__name__}-{cov_table}({len(features)})",
        )
    )
    features = [
        "ao",
        "cmo",
        "cci",
        "connors_rsi",
        "dpo",
        "stoch",
        "rsi",
        "stc",
        "smi",
        "stoch_rsi",
        "trix",
        "ultimate",
        "williams_r",
    ]
    cov_table = "ta_oscillators_view"
    futures.append(
        client.submit(
            covar_metric,
            anchor_symbol,
            anchor_df,
            cov_table,
            features,
            dates,
            min_count,
            args,
            sem,
            locks,
            key=f"{covar_metric.__name__}-{cov_table}({len(features)})",
        )
    )
    features = [
        "pivots",
        "fractal",
    ]
    cov_table = "ta_other_price_patterns_view"
    futures.append(
        client.submit(
            covar_metric,
            anchor_symbol,
            anchor_df,
            cov_table,
            features,
            dates,
            min_count,
            args,
            sem,
            locks,
            key=f"{covar_metric.__name__}-{cov_table}({len(features)})",
        )
    )
    features = [
        "bollinger",
        "donchian",
        "fcb",
        "keltner",
        "ma_envelopes",
        "pivot_points",
        "rolling_pivots",
        "starc_bands",
        "stdev_channels",
    ]
    cov_table = "ta_price_channel_view"
    futures.append(
        client.submit(
            covar_metric,
            anchor_symbol,
            anchor_df,
            cov_table,
            features,
            dates,
            min_count,
            args,
            sem,
            locks,
            key=f"{covar_metric.__name__}-{cov_table}({len(features)})",
        )
    )
    features = [
        "atr",
        "bop",
        "chop",
        "stdev",
        "roc",
        "roc2",
        "pmo",
        "tsi",
        "ulcer_index",
    ]
    cov_table = "ta_price_characteristics_view"
    futures.append(
        client.submit(
            covar_metric,
            anchor_symbol,
            anchor_df,
            cov_table,
            features,
            dates,
            min_count,
            args,
            sem,
            locks,
            key=f"{covar_metric.__name__}-{cov_table}({len(features)})",
        )
    )
    features = [
        "fisher_transform",
        "heikin_ashi",
        "zig_zag",
    ]
    cov_table = "ta_price_transforms_view"
    futures.append(
        client.submit(
            covar_metric,
            anchor_symbol,
            anchor_df,
            cov_table,
            features,
            dates,
            min_count,
            args,
            sem,
            locks,
            key=f"{covar_metric.__name__}-{cov_table}({len(features)})",
        )
    )
    features = [
        "atr",
        "aroon",
        "adx",
        "elder_ray",
        "gator",
        "hurst",
        "ichimoku",
        "macd",
        "super_trend",
        "vortex",
        "alligator",
    ]
    cov_table = "ta_price_trends_view"
    futures.append(
        client.submit(
            covar_metric,
            anchor_symbol,
            anchor_df,
            cov_table,
            features,
            dates,
            min_count,
            args,
            sem,
            locks,
            key=f"{covar_metric.__name__}-{cov_table}({len(features)})",
        )
    )
    features = [
        "chandelier",
        "parabolic_sar",
        "volatility_stop",
    ]
    cov_table = "ta_stop_reverse_view"
    futures.append(
        client.submit(
            covar_metric,
            anchor_symbol,
            anchor_df,
            cov_table,
            features,
            dates,
            min_count,
            args,
            sem,
            locks,
            key=f"{covar_metric.__name__}-{cov_table}({len(features)})",
        )
    )
    features = [
        "adl",
        "cmf",
        "chaikin_osc",
        "force_index",
        "kvo",
        "mfi",
        "obv",
        "pvo",
    ]
    cov_table = "ta_volume_based_view"
    futures.append(
        client.submit(
            covar_metric,
            anchor_symbol,
            anchor_df,
            cov_table,
            features,
            dates,
            min_count,
            args,
            sem,
            locks,
            key=f"{covar_metric.__name__}-{cov_table}({len(features)})",
        )
    )

    # TODO prep options

    # TODO CPI, PPI
    # TODO car sales
    # TODO electricity consumption
    # TODO exports and imports
    # TODO commodity prices: oil, copper, aluminum, coal, gold, etc.
    # TODO cash inflow


def univariate_baseline(anchor_df, hps_id, args):
    global random_seed, client, default_params
    df = anchor_df[["ds", "y"]]
    df_future = client.scatter(df)

    return client.submit(
        log_metrics_for_hyper_params,
        args.symbol,
        df_future,
        default_params,
        args.epochs,
        random_seed,
        select_device(
            args.accelerator,
            getattr(args, "gpu_util_threshold", None),
            getattr(args, "gpu_ram_threshold", None),
        ),
        0,
        hps_id,
        args.early_stopping,
        args.infer_holiday,
    )


def _covar_cutoff_date(symbol):
    global alchemyEngine, args
    with alchemyEngine.connect() as conn:
        match args.model:
            case "NeuralProphet":
                query = (
                    "SELECT max(ts_date) FROM neuralprophet_corel where symbol=:symbol"
                )
                result = conn.execute(text(query), {"symbol": symbol})
                return result.fetchone()[0]
            case _:
                query = "SELECT max(ts_date) FROM paired_correlation where symbol=:symbol and model=:model"
                result = conn.execute(
                    text(query), {"symbol": symbol, "model": args.model}
                )
                return result.fetchone()[0]


def _hps_cutoff_date(symbol, model, method):
    global alchemyEngine
    with alchemyEngine.connect() as conn:
        query = """
            SELECT max(ts_date) 
            FROM hps_sessions 
            WHERE symbol=:symbol 
                AND model=:model 
                AND (method=:method OR method is null)
        """
        result = conn.execute(
            text(query),
            {
                "symbol": symbol,
                "model": model,
                "method": method,
            },
        )
        return result.fetchone()[0]


def _get_cutoff_date(args):
    global logger
    resume = args.resume

    today = datetime.date.today()

    if not resume:
        return today

    covar_cutoff = today
    hps_cutoff = today

    if not args.hps_only:
        cutoff_date = _covar_cutoff_date(args.symbol)
        if cutoff_date is not None:
            covar_cutoff = cutoff_date
    if not args.covar_only:
        cutoff_date = _hps_cutoff_date(args.symbol, args.model, args.method)
        if cutoff_date is not None:
            hps_cutoff = cutoff_date

    # return the smallest date between covar_cutoff and hps_cutoff
    return min(covar_cutoff, hps_cutoff)


def get_hps_session(symbol, model, cutoff_date, resume, timesteps):
    global alchemyEngine

    if resume:
        query = """
            select max(id) from (
                select max(id) id
                from hps_sessions
                where symbol = :symbol
                    and model = :model
                    and ts_date = :ts_date
                union all
                select max(id) id
                from hps_sessions
                where symbol = :symbol
                    and model = :model
                    and search_space is null
            )
        """
        with alchemyEngine.connect() as conn:
            result = conn.execute(
                text(query),
                {
                    "symbol": symbol,
                    "ts_date": cutoff_date,
                    "model": model,
                },
            )
            max_id = result.fetchone()[0]
            if max_id is not None:
                query_covar_set_id = """
                    select covar_set_id
                    from hps_sessions
                    where id = :id
                """
                result = conn.execute(
                    text(query_covar_set_id),
                    {
                        "id": max_id,
                    },
                )
                covar_set_id = result.fetchone()[0]
                return max_id, covar_set_id

    with alchemyEngine.begin() as conn:
        result = conn.execute(
            text(
                """
                INSERT INTO hps_sessions (symbol, model, ts_date, timesteps) 
                VALUES (:symbol, :model, :ts_date, :timesteps)
                RETURNING id
                """
            ),
            {
                "symbol": symbol,
                "model": model,
                "ts_date": cutoff_date,
                "timesteps": timesteps,
            },
        )
        return result.fetchone()[0], None


def main(_args):
    global client, logger, futures, alchemyEngine, random_seed, args
    args = _args
    t_start = time.time()
    try:
        init(args)

        cutoff_date = _get_cutoff_date(args)
        anchor_df, anchor_table = load_anchor_ts(
            args.symbol, args.timestep_limit, alchemyEngine, cutoff_date
        )
        cutoff_date = anchor_df["ds"].max().strftime("%Y-%m-%d")

        # TODO: make use of the returned covar_set_id to resume?
        hps_id, _ = get_hps_session(args.symbol, args.model, cutoff_date, args.resume)
        logger.info(
            "HPS session ID: %s, Cutoff date: %s, CovarSet ID: %s",
            hps_id,
            cutoff_date,
            covar_set_id,
        )

        futures.append(univariate_baseline(anchor_df, hps_id, args))

        if not args.hps_only:
            t1_start = time.time()
            prep_covar_baseline_metrics(anchor_df, anchor_table, args)
            if not args.covar_only:
                ## wait for all tasks to be completed before restarting
                await_futures(futures)
                client.restart()
            logger.info(
                "%s covariate baseline metric computation completed. Time taken: %s seconds",
                args.symbol,
                time.time() - t1_start,
            )

        if not args.covar_only:
            t2_start = time.time()
            df, covar_set_id, ranked_features = augment_anchor_df_with_covars(
                anchor_df, args, alchemyEngine, logger, cutoff_date
            )
            df_future = client.scatter(df)
            ranked_features_future = client.scatter(ranked_features)
            if args.method == "gs":
                grid_search(df_future, covar_set_id, hps_id, ranked_features_future)
            elif args.method == "bayesopt":
                bayesopt(df_future, covar_set_id, hps_id, ranked_features_future)
            else:
                raise ValueError(f"unsupported search method: {args.method}")
            logger.info(
                "%s hyper-parameter search completed. Time taken: %s seconds",
                args.symbol,
                time.time() - t2_start,
            )

        await_futures(futures)
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
        "--max_covars",
        action="store",
        type=int,
        default=100,
        help=(
            "Limit the maximum number of top-covariates to be included for training and prediction. "
            "If it's less than 1, we'll use all covariates with loss_val less than univariate baseline. "
            "Defaults to 100."
        ),
    )
    group2.add_argument(
        "--covar_set_id",
        action="store",
        type=int,
        default=None,
        help=(
            "Covariate set ID corresponding to the covar_set table. "
            "If not set, the grid search will look for latest max_covars covariates with loss_val less than univariate baseline "
            "as found in the neuralprophet_corel table, which could be non-static."
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
        help=(
            "Limit the ratio of NaN (missing data) in covariates. "
            "Only those with NaN rate lower than the limit ratio can be selected during multivariate grid searching."
            "Defaults to 0.5%."
        ),
    )
    parser.add_argument(
        "--accelerator", action="store_true", help="Use accelerator automatically"
    )
    parser.add_argument(
        "--infer_holiday",
        action="store_true",
        help=(
            "Infer holiday region based on anchor symbol's nature, "
            "which will be utilized during covariate-searching and grid-search."
        ),
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
