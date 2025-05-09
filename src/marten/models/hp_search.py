import os

# OPENBLAS_NUM_THREADS = 1
# os.environ["OPENBLAS_NUM_THREADS"] = f"{OPENBLAS_NUM_THREADS}"

import time
import random
import datetime
import argparse
import json
import math
import multiprocessing
import pandas as pd
import numpy as np
import uuid
import psutil
import torch

from dotenv import load_dotenv

from dask.distributed import (
    get_worker,
    worker_client,
    wait,
)

# import ray

from marten.models.base_model import BaseModel
from marten.utils.database import get_database_engine, columns_with_prefix
from marten.utils.logger import get_logger
from marten.utils.worker import (
    get_results,
    await_futures,
    init_client,
    num_workers,
    hps_task_callback,
    restart_all_workers,
)
from marten.utils.softs import SOFTSPredictor, baseline_config
from marten.utils.neuralprophet import select_topk_features
from marten.utils.softs import is_large_model

# from marten.utils.system import init_cpu_core_id
from marten.utils.trainer import (
    select_device,
    select_randk_covars,
    # get_accelerator_locks,
)
from marten.utils.worker import scale_cluster_and_wait
from marten.models.worker_func import (
    fit_with_covar,
    log_metrics_for_hyper_params,
    validate_hyperparams,
    LOSS_CAP,
    count_topk_hp,
    impute,
)
from marten.features.build_features import extract_features

from sqlalchemy import text

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
    load_dotenv()  # take environment variables from .env.

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
            from marten.models.nf_tsmixerx import TSMixerxModel

            model = TSMixerxModel()
        case _:
            model = None


def _init_local_resource():
    global logger, alchemyEngine
    # NeuralProphet: Disable logging messages unless there is an error
    set_log_level("ERROR")

    if alchemyEngine is None:
        DB_USER = os.getenv("DB_USER")
        DB_PASSWORD = os.getenv("DB_PASSWORD")
        DB_HOST = os.getenv("DB_HOST")
        DB_PORT = os.getenv("DB_PORT")
        DB_NAME = os.getenv("DB_NAME")

        db_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        alchemyEngine = get_database_engine(db_url)

    if logger is None:
        logger = get_logger(__name__)


def load_anchor_ts(
    symbol, limit, alchemyEngine, cutoff_date=None, anchor_table: str = None
):
    ## support arbitrary types of symbol (could be from different tables, with different features available)
    tbl_cols_dict = {
        "index_daily_em_view": (
            "date DS, change_rate y, vol_change_rate, amt_change_rate, open, close, high, low, volume, amount, "
            "open_preclose_rate, high_preclose_rate, low_preclose_rate"
        ),
        "fund_etf_daily_em_view": (
            "date DS, change_rate y, vol_change_rate, open, close, high, low, volume, "
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
            "open_preclose_rate, high_preclose_rate, low_preclose_rate, vol_change_rate"
        ),
    }
    if anchor_table is None or anchor_table == "unspecified":
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
    elif not anchor_table.endswith("_view"):
        anchor_table = f"{anchor_table}_view"

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
    symbol_table,
    dates,
    table,
    feature,
    ts_date,
    min_count,
):
    worker = get_worker()
    alchemyEngine, logger = worker.alchemyEngine, worker.logger

    params = {
        "table": table,
        "anchor_symbol": anchor_symbol,
        "feature": feature,
        "ts_date": ts_date,
        "start_date": min(dates),
        "end_date": max(dates),
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
                    AND pc.symbol_table = %(symbol_table)s
                    AND pc.cov_table = %(table)s
                    AND pc.feature = %(feature)s
                    AND pc.ts_date = %(ts_date)s
            """
            params["model"] = model
            params["symbol_table"] = symbol_table

    # get a list of symbols from the given table, of which metrics are not recorded yet
    if table.startswith("ta_"):
        symbol_col = "t.table_symbol"
        exclude = "1=1"
        column_names = columns_with_prefix(alchemyEngine, table, feature)
        notnull = "(" + " or ".join([f"t.{c} is not null" for c in column_names]) + ")"
        group_by = f"group by {symbol_col}"
    else:
        symbol_col = "t.symbol"
        exclude = "t.symbol <> %(anchor_symbol)s"
        notnull = f"t.{feature} is not null"
        group_by = f"group by {symbol_col}"

    orig_table = table[:-5] if table.endswith("_view") else table

    query = f"""
        WITH existing_cov_symbols AS ({existing_cov_symbols}),
        eligible_symbols AS (
            SELECT
                {symbol_col} symbol,
                COUNT(*) AS num
            FROM
                {orig_table} t
            WHERE
                {notnull}
                AND {exclude}
                AND t.date BETWEEN %(start_date)s and %(end_date)s
            {group_by}
            HAVING COUNT(*) >= %(min_count)s
        )
        SELECT
            es.symbol,
            es.num
        FROM
            eligible_symbols es
        LEFT JOIN existing_cov_symbols ecs ON es.symbol = ecs.cov_symbol
        WHERE
            ecs.cov_symbol IS NULL
    """

    with alchemyEngine.connect() as conn:
        cov_symbols = pd.read_sql(
            query, conn, params=params, dtype={"symbol": "string", "num": "int32"}
        )

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
    anchor_symbol, anchor_df, cov_table, features, args, cutoff_date
):
    global client, futures, logger, alchemyEngine

    # remove feature elements already exists in the covar table.
    features = _remove_measured_features(
        alchemyEngine,
        args.model,
        anchor_symbol,
        args.symbol_table,
        cov_table,
        features,
        cutoff_date,
    )

    if not features:
        return

    cpu_count = psutil.cpu_count(logical=False)

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
            # select_device(
            #     args.accelerator,
            #     getattr(args, "gpu_util_threshold", None),
            #     getattr(args, "gpu_ram_threshold", None),
            # ),
            "auto",
            args.early_stopping,
            args.infer_holiday,
        )
        futures.append(future)
        # await_futures(futures, False)
        if len(futures) > cpu_count:
            done, undone = wait(futures, return_when="FIRST_COMPLETED")
            for f in done:
                get_results(done)
            futures = list(undone)


def _pair_covar_metrics(
    # client,
    anchor_symbol,
    anchor_df,
    cov_table,
    cov_symbols,
    feature,
    min_date,
    args,
):
    worker = get_worker()
    logger = worker.logger
    cpu_count = psutil.cpu_count()
    covar_futures = []

    for symbol in cov_symbols:
        logger.debug(
            "submitting fit_with_covar: %s @ %s.%s",
            symbol,
            cov_table,
            feature,
        )
        # NOTE: the worker_client() must be within the for loop rather than outside,
        # to avoid scheculer connection lost error
        with worker_client() as client:
            covar_futures.append(
                client.submit(
                    fit_with_covar,
                    anchor_symbol,
                    anchor_df,
                    cov_table,
                    symbol,
                    min_date,
                    args.random_seed,
                    feature,
                    # select_device(
                    #     args.accelerator,
                    #     getattr(args, "gpu_util_threshold", None),
                    #     getattr(args, "gpu_ram_threshold", None),
                    # ),
                    "auto",
                    args.early_stopping,
                    args.infer_holiday,
                    key=f"{fit_with_covar.__name__}({cov_table})-{uuid.uuid4().hex}",
                    # priority=p_order,
                )
            )
            # if too much pending task, then slow down for the tasks to be digested
            # await_futures(covar_futures, False, multiplier=0.5, max_delay=300)
            if len(covar_futures) > cpu_count * 3:
                # with worker_client():
                try:
                    done, undone = wait(covar_futures, return_when="FIRST_COMPLETED")
                    if len(done) + len(undone) != len(covar_futures):
                        logger.warning(
                            "done(%s) + undone(%s) != total(%s)",
                            len(done),
                            len(undone),
                            len(covar_futures),
                        )
                    get_results(done)
                    del done
                except Exception as e:
                    logger.exception(
                        "failed to wait covar_futures: %s", e, exc_info=True
                    )
                    client.dump_cluster_state(
                        "cluster_state_dump", write_from_scheduler=True
                    )
                    raise e

                covar_futures = list(undone)

            # for f in done:
            #     get_result(f)
        # wait(covar_futures)
    with worker_client():
        while len(covar_futures) > 0:
            done, undone = wait(covar_futures)
            get_results(done)
            del done
            covar_futures = list(undone)
    # await_futures(covar_futures)


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
                    AND t1.symbol_table = t2.symbol_table
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
                    PARTITION BY t1.symbol, t1.symbol_table, t1.cov_table, t1.cov_symbol, t1.cov_feature
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
    symbol_table,
    nan_threshold=None,
    cov_table=None,
    feature=None,
    cutoff_date=None,
    model=None,
):
    global logger
    params = {
        "anchor_symbol": anchor_symbol,
        "symbol_table": symbol_table,
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
                    and symbol_table = %(symbol_table)s
                    and cov_symbol = %(anchor_symbol)s
                    and feature = 'y'
                    and ts_date <= %(ts_date)s
                    order by ts_date desc
                    limit 1
            """
            query = f"""
                select DISTINCT ON (ts_date, loss_val, nan_count, cov_table, cov_symbol)
                    cov_symbol, cov_table, feature, loss_val
                from
                    paired_correlation
                where
                    model = %(model)s
                    and symbol = %(anchor_symbol)s
                    and symbol_table = %(symbol_table)s
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
    with alchemyEngine.connect() as conn:
        df = pd.read_sql(
            query,
            conn,
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

    params = {"symbol": anchor_symbol, "num": len(df)}
    values_clause = []
    for i, row in enumerate(df.itertuples(index=False, name=None)):
        params.update(
            {
                f"cov_symbol_{i}": row[0],
                f"cov_table_{i}": row[1],
                f"cov_feature_{i}": row[2],
            }
        )
        values_clause.append(f"(:cov_symbol_{i}, :cov_table_{i}, :cov_feature_{i})")

    values_sql = ",\n    ".join(values_clause)

    query = f"""
        WITH df_values (cov_symbol, cov_table, cov_feature) AS (
            VALUES 
                {values_sql}
        )
        SELECT cs.id
        FROM covar_set cs
        LEFT JOIN df_values dv ON 
            cs.cov_symbol = dv.cov_symbol AND 
            cs.cov_table = dv.cov_table AND 
            cs.cov_feature = dv.cov_feature
        WHERE 
            cs.symbol = :symbol
        GROUP BY cs.id
        HAVING 
            COUNT(*) = :num
            AND
            COUNT(dv.cov_symbol) = :num
        ORDER BY cs.id DESC
    """

    with alchemyEngine.begin() as conn:
        # need to use transaction for inserting new covar_set records.
        # check if the same set of covar features exists in `covar_set` table. If so, reuse the same set_id.
        result = conn.execute(text(query), params)
        # matching_ids = result.fetchall()
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
            table_df = df.rename(
                columns={"feature": "cov_feature", "loss_val": "score"}
            )
            table_df["symbol_table"] = symbol_table
            table_df["symbol"] = anchor_symbol
            table_df["id"] = covar_set_id
            table_df.to_sql("covar_set", con=conn, if_exists="append", index=False)

    return df, covar_set_id


def _load_covar_feature(
    anchor_table, anchor_symbol, cov_table, feature, symbols, start_date, end_date
):
    worker = get_worker()
    alchemyEngine = worker.alchemyEngine
    params = {"start_date": start_date, "end_date": end_date}
    match cov_table:
        case "bond_metrics_em" | "bond_metrics_em_view":
            query = f"""
                SELECT 'bond' ID, date DS, {feature} y
                FROM {cov_table}
                and date between %(start_date)s and %(end_date)s
            """
            table_feature_df = pd.read_sql(
                query, alchemyEngine, params=params, parse_dates=["ds"]
            )
        case "currency_boc_safe_view":
            query = f"""
                SELECT 'currency_exchange' ID, date DS, {feature} y
                FROM {cov_table}
                and date between %(start_date)s and %(end_date)s
            """
            table_feature_df = pd.read_sql(
                query, alchemyEngine, params=params, parse_dates=["ds"]
            )
        case "ts_features_view":
            table_symbols = [tuple(s.split("::", 1)) for s in symbols]
            # Ensure 'table' and 'symbol' are correctly quoted to prevent SQL injection
            values_list = ", ".join(
                [
                    "(%(cov_table_{0})s, %(cov_symbol_{0})s)".format(idx)
                    for idx in range(len(table_symbols))
                ]
            )
            for idx, (table_name, symbol) in enumerate(table_symbols):
                params[f"cov_table_{idx}"] = table_name
                params[f"cov_symbol_{idx}"] = symbol
            params["symbol_table"] = anchor_table
            params["symbol"] = anchor_symbol
            params["feature"] = feature
            # Construct the query using named parameters
            query = f"""
                SELECT symbol_table || '::' || symbol AS ID, date AS DS, value as {feature}
                FROM ts_features_view
                WHERE 
                symbol_table = %(symbol_table)s
                and symbol = %(symbol)s
                and feature = %(feature)s
                and (cov_table, cov_symbol) IN ({values_list})
                and date between %(start_date)s and %(end_date)s
            """
            table_feature_df = pd.read_sql(
                query, alchemyEngine, params=params, parse_dates=["ds"]
            )
        case _ if cov_table.startswith("ta_"):  # handle technical indicators table
            column_names = columns_with_prefix(alchemyEngine, cov_table, feature)
            table_symbols = [tuple(s.split("::")) for s in symbols]
            select_cols = ", ".join(column_names)
            # Ensure 'table' and 'symbol' are correctly quoted to prevent SQL injection
            values_list = ", ".join(
                [
                    "(%(table_{0})s, %(symbol_{0})s)".format(idx)
                    for idx in range(len(table_symbols))
                ]
            )
            for idx, (table_name, symbol) in enumerate(table_symbols):
                params[f"table_{idx}"] = table_name
                params[f"symbol_{idx}"] = symbol
            # Construct the query using named parameters
            query = f"""
                SELECT "table" || '::' || symbol AS ID, date AS DS, {select_cols}
                FROM {cov_table}
                WHERE ("table", symbol) IN ({values_list})
                and date between %(start_date)s and %(end_date)s
            """
            # if sem:
            #     with sem:
            #         table_feature_df = pd.read_sql(
            #             query, alchemyEngine, params=params, parse_dates=["ds"]
            #         )
            # else:
            table_feature_df = pd.read_sql(
                query, alchemyEngine, params=params, parse_dates=["ds"]
            )
        case _:
            query = f"""
                SELECT symbol ID, date DS, {feature} y
                FROM {cov_table}
                where symbol in %(symbols)s
                and date between %(start_date)s and %(end_date)s
            """
            params["symbols"] = tuple(symbols)
            table_feature_df = pd.read_sql(
                query, alchemyEngine, params=params, parse_dates=["ds"]
            )
    return table_feature_df


def augment_anchor_df_with_covars(df, args, alchemyEngine, logger, cutoff_date):
    global client
    # date_col = "ds" if args.model == "NeuralProphet" else "date"
    merged_df = df[["ds", "y"]].copy()
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
            args.symbol_table,
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
        f'{r["feature"]}::{r["cov_table"]}::{r["cov_symbol"]}'
        for _, r in covars_df.iterrows()
    ]

    # covars_df contain these columns: cov_symbol, cov_table, feature, loss_val
    by_table_feature = covars_df.groupby(["cov_table", "feature"])
    futures = []
    start_date = merged_df["ds"].min().strftime("%Y-%m-%d")
    end_date = merged_df["ds"].max().strftime("%Y-%m-%d")
    for group1, sdf1 in by_table_feature:
        ## load covariate time series from different tables and/or features
        cov_table = group1[0]
        feature = group1[1]
        futures.append(
            client.submit(
                _load_covar_feature,
                args.symbol_table,
                args.symbol,
                cov_table,
                feature,
                sdf1["cov_symbol"],
                start_date,
                end_date,
            )
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
        for group2, sdf2 in grouped:  # group2 = symbol
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
                for col in [c for c in sdf2.columns if c.startswith(f"{feature}_")]:
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


def _search_space(model_name, model, max_covars, topk_covars):
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
            return model.search_space(
                topk_covars=topk_covars if topk_covars > 0 else max_covars,
                max_covars=max_covars,
            )
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
    model_name, anchor_symbol, symbol_table, covar_set_id, hps_id, feat_size
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
                    and symbol_table = :symbol_table
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
                "symbol_table": symbol_table,
                "covar_set_id": covar_set_id,
                "hps_id": hps_id,
                # "limit": limit,
            },
        )

        tuples = []
        for row in results:
            # param_dict = json.loads(row[0], object_hook=hp_deserializer)
            param_dict = row[0]

            if "num_covars" in param_dict:
                param_dict.pop("num_covars")

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
):
    global logger, client, model

    cuda_count = torch.cuda.device_count()

    @scheduler.custom(n_jobs=n_jobs)
    def objective(params_batch):
        nonlocal cuda_count
        num_gpu = cuda_count
        jobs = []
        t1 = time.time()
        nworker = num_workers(False)
        client.set_metadata(["workload_info", "total"], len(params_batch))
        client.set_metadata(["workload_info", "workers"], nworker)
        client.set_metadata(["workload_info", "finished"], 0)
        tasks = []
        cpu_task_pos = 0
        
        for params in params_batch:
            new_df = df.copy()
            priority = 1
            if "topk_covar" in params:
                if "covar_dist" in params:
                    new_df = select_randk_covars(
                        new_df,
                        ranked_features,
                        params["covar_dist"],
                        params["topk_covar"],
                    )
                else:
                    new_df = select_topk_features(
                        new_df, ranked_features, params["topk_covar"]
                    )
            params["num_covars"] = len(
                [c for c in new_df.columns if c not in ("ds", "y")]
            )
            if model and model.trainable_on_cpu(**params):
                tasks.insert(cpu_task_pos, (new_df, params, priority + 10))
            else:
                if num_gpu > 0:
                    tasks.insert(0, (new_df, params, priority))
                    cpu_task_pos += 1
                    num_gpu -= 1
                else:
                    tasks.append((new_df, params, priority))

        for i, (new_df, params, priority) in enumerate(tasks):
            future = client.submit(
                validate_hyperparams,
                args,
                new_df,
                covar_set_id,
                hps_id,
                params,
                resources={"POWER": power_demand(args, params)},
                retries=1,
                key=f"{validate_hyperparams.__name__}-{uuid.uuid4().hex}",
                priority=priority,
            )
            future.add_done_callback(hps_task_callback)
            jobs.append(future)
            if i < nworker:
                time.sleep(random.uniform(1, 5))
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
        for param in params:
            param.pop("num_covars", None)
        logger.info("Elapsed: %s, Successful results: %s", elapsed, len(results))
        # restart client here to free up memory
        if args.restart_workers:
            restart_all_workers(client)
            # client.restart()
        return params, loss

    warmstart_tuples = None
    if resume:
        warmstart_tuples = preload_warmstart_tuples(
            args.model,
            args.symbol,
            args.symbol_table,
            covar_set_id,
            hps_id,
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
            i > 0 or args.resume.lower() != "none",
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


def _remove_measured_features(
    alchemyEngine, model, anchor_symbol, symbol_table, cov_table, features, ts_date=None
):
    params = {
        "symbol": anchor_symbol,
        "symbol_table": symbol_table,
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
                    and symbol_table = %(symbol_table)s
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
    anchor_symbol,
    anchor_df,
    cov_table,
    features,
    dates,
    min_count,
    args,
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
            alchemyEngine,
            args.model,
            anchor_symbol,
            args.symbol_table,
            cov_table,
            features,
            cutoff_date,
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

    num_symbols = 0
    # with worker_client() as client:
    for feature in features:
        match cov_table:
            case "bond_metrics_em" | "bond_metrics_em_view":
                # construct a dummy cov_symbols dataframe with `symbol` column and the value 'bond'.
                _pair_covar_metrics(
                    # client,
                    anchor_symbol,
                    anchor_df,
                    cov_table,
                    ["bond"],
                    feature,
                    min_date,
                    args,
                )
                num_symbols += 1
            case "currency_boc_safe_view":
                _pair_covar_metrics(
                    # client,
                    anchor_symbol,
                    anchor_df,
                    cov_table,
                    ["currency_exchange"],
                    feature,
                    min_date,
                    args,
                )
                num_symbols += 1
            case _:
                cov_symbols, feature = covar_symbols_from_table(
                    args.model,
                    anchor_symbol,
                    args.symbol_table,
                    dates,
                    cov_table,
                    feature,
                    cutoff_date,
                    min_count,
                )
                _pair_covar_metrics(
                    # client,
                    anchor_symbol,
                    anchor_df,
                    cov_table,
                    cov_symbols,
                    feature,
                    min_date,
                    args,
                )
                num_symbols += len(cov_symbols)

    # await_futures(covar_futures)
    # with worker_client():
    #     while len(covar_futures) > 0:
    #         done, undone = wait(covar_futures)
    #         get_results(done)
    #         del done
    #         covar_futures = list(undone)

    logger.info(
        "finished covar_metric for %s features in %s, total covar symbols: %s",
        len(features),
        cov_table,
        num_symbols,
    )
    return num_symbols


def prep_covar_baseline_metrics(anchor_df, anchor_table, args):
    global random_seed, client, futures

    anchor_symbol = args.symbol

    # min_date = anchor_df["ds"].min().strftime("%Y-%m-%d")
    cutoff_date = anchor_df["ds"].max().strftime("%Y-%m-%d")
    min_count = int(len(anchor_df) * (1 - args.nan_limit))
    dates = anchor_df["ds"].dt.date.tolist()

    # if not sem:
    #     max_leases = (
    #         args.resource_intensive_sql_semaphore
    #         if args.resource_intensive_sql_semaphore > 0
    #         else int(os.getenv("RESOURCE_INTENSIVE_SQL_SEMAPHORE", args.min_worker))
    #     )
    #     if max_leases > 0:
    #         dask.config.set({"distributed.scheduler.locks.lease-timeout": "500s"})
    #         sem = Semaphore(
    #             max_leases=max_leases,
    #             name="RESOURCE_INTENSIVE_SQL_SEMAPHORE",
    #         )
    # if not locks:
    #     locks = get_accelerator_locks(0, gpu_leases=2, mps_leases=0, timeout="20s")

    # init_cpu_core_id(alchemyEngine)

    # endogenous features of the anchor time series per se
    endogenous_features = [col for col in anchor_df.columns if col not in ("ds")]
    _pair_endogenous_covar_metrics(
        anchor_symbol,
        anchor_df,
        anchor_table,
        endogenous_features,
        args,
        cutoff_date,
    )

    # _, undone = wait(futures)
    # futures = list(undone)

    # for the rest of exogenous covariates, keep only the core features of anchor_df
    anchor_df = anchor_df[["ds", "y"]].copy()

    table_features = {
        "CN_Index": (
            "index_daily_em_view",
            [
                "change_rate",
                "amt_change_rate",
                "vol_change_rate",
                "open_preclose_rate",
                "high_preclose_rate",
                "low_preclose_rate",
            ],
        ),
        "ETF": (
            "fund_etf_daily_em_view",
            [
                "change_rate",
                "vol_change_rate",
                "turnover_rate",
                "turnover_change_rate",
                "open_preclose_rate",
                "high_preclose_rate",
                "low_preclose_rate",
            ],
        ),
        "Bond": (
            "bond_metrics_em_view",
            [
                "china_yield_2y",
                "china_yield_10y",
                "china_yield_30y",
                "china_yield_spread_10y_2y",
                "us_yield_2y",
                "us_yield_10y",
                "us_yield_30y",
                "us_yield_spread_10y_2y",
                "quantile",
                "performance_benchmark",
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
            ],
        ),
        "Bond_Index": (
            "cn_bond_indices_view",
            [
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
            ],
        ),
        "CN_Bond": (
            "bond_zh_hs_daily_view",
            [
                "change_rate",
                "vol_change_rate",
                "open_preclose_rate",
                "high_preclose_rate",
                "low_preclose_rate",
            ],
        ),
        "US_Index": (
            "us_index_daily_sina_view",
            [
                "change_rate",
                "amt_change_rate",
                "vol_change_rate",
                "open_preclose_rate",
                "high_preclose_rate",
                "low_preclose_rate",
            ],
        ),
        "HK_Index": (
            "hk_index_daily_em_view",
            [
                "change_rate",
                "open_preclose_rate",
                "high_preclose_rate",
                "low_preclose_rate",
            ],
        ),
        "CN_Stock": (
            "stock_zh_a_hist_em_view",
            [
                "change_rate",
                "turnover_rate",
                "turnover_change_rate",
                "open_preclose_rate",
                "high_preclose_rate",
                "low_preclose_rate",
                "vol_change_rate",
            ],
        ),
        "Currency": (
            "currency_boc_safe_view",
            [
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
            ],
        ),
        "SGE_Spot": (
            "spot_hist_sge_view",
            [
                "change_rate",
                "open_preclose_rate",
                "high_preclose_rate",
                "low_preclose_rate",
            ],
        ),
        "Interbank": (
            "interbank_rate_hist_view",
            ["change_rate"],
        ),
        "TA_MA": (
            "ta_ma_view",
            [
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
            ],
        ),
        "TA_Numerical_Analysis": (
            "ta_numerical_analysis_view",
            [
                "slope",
            ],
        ),
        "TA_Oscillators": (
            "ta_oscillators_view",
            [
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
            ],
        ),
        "TA_Other_Price_Patterns": (
            "ta_other_price_patterns_view",
            [
                "pivots",
                "fractal",
            ],
        ),
        "TA_Price_Channel": (
            "ta_price_channel_view",
            [
                "bollinger",
                "donchian",
                "fcb",
                "keltner",
                "ma_envelopes",
                "pivot_points",
                "rolling_pivots",
                "starc_bands",
                "stdev_channels",
            ],
        ),
        "TA_Price_Characteristics": (
            "ta_price_characteristics_view",
            [
                "atr",
                "bop",
                "chop",
                "stdev",
                "roc",
                "roc2",
                "pmo",
                "tsi",
                "ulcer_index",
            ],
        ),
        "TA_Price_Transforms": (
            "ta_price_transforms_view",
            [
                "fisher_transform",
                "heikin_ashi",
                "zig_zag",
            ],
        ),
        "TA_Price_Trends": (
            "ta_price_trends_view",
            [
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
            ],
        ),
        "TA_Stop_Reverse": (
            "ta_stop_reverse_view",
            [
                "chandelier",
                "parabolic_sar",
                "volatility_stop",
            ],
        ),
        "TA_Volume_Based": (
            "ta_volume_based_view",
            [
                "adl",
                "cmf",
                "chaikin_osc",
                "force_index",
                "kvo",
                "mfi",
                "obv",
                "pvo",
            ],
        ),
        "Option_QVIX": (
            "option_qvix_view",
            [
                "open",
                "high",
                "low",
                "close",
            ],
        ),
    }

    # TODO prep options

    # TODO CPI, PPI
    # TODO car sales
    # TODO electricity consumption
    # TODO exports and imports
    # TODO commodity prices: oil, copper, aluminum, coal, gold, etc.
    # TODO cash inflow

    tasks = []
    keys = list(table_features.keys())
    for i in range(0, len(keys)):
        cov_table, features = table_features[keys[i]]
        tasks.append(
            client.submit(
                covar_metric,
                anchor_symbol,
                anchor_df,
                cov_table,
                features,
                dates,
                min_count,
                args,
                # p_order=len(keys) - i,
                # priority=len(keys) - i,
                key=f"{covar_metric.__name__}_{keys[i].lower()}({len(features)})-"
                # + f"{cov_table}({len(features)})_{len(keys) - i}",
                + uuid.uuid4().hex,
            )
        )
        if len(tasks) > 1:
            done, undone = wait(tasks, return_when="FIRST_COMPLETED")
            get_results(done)
            del done
            tasks = list(undone)

    futures.extend(tasks)


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
                query = "SELECT max(ts_date) FROM paired_correlation where symbol=:symbol and model=:model and symbol_table=:symbol_table"
                result = conn.execute(
                    text(query),
                    {
                        "symbol": symbol,
                        "model": args.model,
                        "symbol_table": args.symbol_table,
                    },
                )
                return result.fetchone()[0]


def _hps_cutoff_date(symbol, symbol_table, model, method):
    global alchemyEngine
    with alchemyEngine.connect() as conn:
        query = """
            SELECT max(ts_date) 
            FROM hps_sessions 
            WHERE symbol=:symbol 
                AND symbol_table=:symbol_table
                AND model=:model 
                AND (method=:method OR method is null)
        """
        result = conn.execute(
            text(query),
            {
                "symbol": symbol,
                "symbol_table": symbol_table,
                "model": model,
                "method": method,
            },
        )
        return result.fetchone()[0]


def _get_cutoff_date(args):
    global logger
    resume = args.resume

    today = datetime.date.today()

    if resume.lower() == "none":
        return today

    covar_cutoff = today
    hps_cutoff = today

    if not args.hps_only:
        cutoff_date = _covar_cutoff_date(args.symbol)
        if cutoff_date is not None:
            covar_cutoff = cutoff_date
    if not args.covar_only:
        cutoff_date = _hps_cutoff_date(
            args.symbol, args.symbol_table, args.model, args.method
        )
        if cutoff_date is not None:
            hps_cutoff = cutoff_date

    # return the smallest date between covar_cutoff and hps_cutoff
    return min(covar_cutoff, hps_cutoff)


def get_hps_session(symbol, symbol_table, model, cutoff_date, resume, timesteps):
    global alchemyEngine

    if resume:
        query = """
            select max(id) from (
                select max(id) id
                from hps_sessions
                where symbol = :symbol
                    and symbol_table = :symbol_table
                    and model = :model
                    and ts_date = :ts_date
                union all
                select max(id) id
                from hps_sessions
                where symbol = :symbol
                    and symbol_table = :symbol_table
                    and model = :model
                    and search_space is null
            )
        """
        with alchemyEngine.connect() as conn:
            result = conn.execute(
                text(query),
                {
                    "symbol": symbol,
                    "symbol_table": symbol_table,
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
                INSERT INTO hps_sessions (symbol, symbol_table, model, ts_date, timesteps) 
                VALUES (:symbol, :symbol_table, :model, :ts_date, :timesteps)
                RETURNING id
                """
            ),
            {
                "symbol": symbol,
                "symbol_table": symbol_table,
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
        hps_id, _ = get_hps_session(
            args.symbol, args.model, cutoff_date, args.resume.lower() != "none"
        )
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
                restart_all_workers(client)
                # client.restart()
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


def init_hps(_model, symbol, _args, _client, _alchemyEngine, _logger):
    global logger, alchemyEngine, args, client, model
    # worker = get_worker()
    # alchemyEngine, logger = worker.alchemyEngine, worker.logger

    _args.symbol = symbol
    _args.hps_only = False
    _args.covar_only = False
    _args.infer_holiday = True
    _args.method = "fast_bayesopt"

    logger = _logger
    alchemyEngine = _alchemyEngine
    args = _args

    client = _client
    model = _model

    return args


def _univariate_default_hp(model, client, anchor_df, args, hps_id):
    df = anchor_df[["ds", "y"]].copy()
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
        # select_device(
        #     args.accelerator,
        #     getattr(args, "gpu_util_threshold", None),
        #     getattr(args, "gpu_ram_threshold", None),
        # ),
        "auto",
        0,
        hps_id,
        args.early_stopping,
        args.infer_holiday,
    ).result()


def min_covar_loss_val(alchemyEngine, model, symbol, symbol_table, ts_date):
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
                                and symbol_table = :symbol_table
                                and ts_date = :ts_date
                        """
                    ),
                    {
                        "model": model,
                        "symbol": symbol,
                        "symbol_table": symbol_table,
                        "ts_date": ts_date,
                    },
                )
        return result.fetchone()[0]


def covars_and_search(model, client, symbol, alchemyEngine, logger, args):
    global futures

    args = init_hps(model, symbol, args, client, alchemyEngine, logger)
    cutoff_date = _get_cutoff_date(args)
    anchor_df, anchor_table = load_anchor_ts(
        args.symbol, args.timestep_limit, alchemyEngine, cutoff_date, args.symbol_table
    )
    cutoff_date = anchor_df["ds"].max().strftime("%Y-%m-%d")

    hps_id, covar_set_id = get_hps_session(
        args.symbol,
        args.symbol_table,
        args.model,
        cutoff_date,
        args.resume.lower() != "none",
        len(anchor_df),
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

    min_covar_loss = min_covar_loss_val(
        alchemyEngine, args.model, symbol, args.symbol_table, cutoff_date
    )
    min_covar_loss = min_covar_loss if min_covar_loss is not None else LOSS_CAP

    base_loss = min(float(univ_loss) * args.loss_quantile, min_covar_loss)

    # if in resume mode, check if the topk HP is present, and further check if prediction is already conducted.
    topk_count = count_topk_hp(alchemyEngine, args.model, hps_id, base_loss)
    if args.resume.lower() != "none" and topk_count >= args.topk:
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

    logger.info("Scaling dask cluster to %s", args.max_worker)
    client.cluster.scale(args.max_worker)
    scale_cluster_and_wait(client, args.max_worker)

    if args.resume in ("none", "covar"):
        # run covariate loss calculation in batch
        logger.info("Starting covariate loss calculation")
        t1_start = time.time()
        prep_covar_baseline_metrics(anchor_df, anchor_table, args)
        # logger.info("waiting dask futures: %s", len(hps.futures))
        while len(futures) > 0:
            done, undone = wait(futures)
            get_results(done)
            del done
            futures = list(undone)
        logger.info(
            "%s covariate baseline metric computation completed. Time taken: %s seconds",
            args.symbol,
            round(time.time() - t1_start, 3),
        )

        if args.extract_extra_features:
            worker_size = int(math.pow(args.min_worker * args.max_worker, 0.6))
            logger.info("Scaling down dask cluster to %s", worker_size)
            scale_cluster_and_wait(client, worker_size)

            t1_start = time.time()
            logger.info("Starting feature engineering and extraction")
            futures = extract_features(
                client, alchemyEngine, symbol, anchor_df, anchor_table, args
            )
            while len(futures) > 0:
                done, undone = wait(futures)
                get_results(done)
                del done
                futures = list(undone)
            logger.info(
                "%s feature extraction completed. Time taken: %s seconds",
                args.symbol,
                round(time.time() - t1_start, 3),
            )

    # scale-in to preserve more memory for hps
    if args.model != "NeuralProphet":
        worker_size = int(math.sqrt(args.min_worker * args.max_worker))
        # worker_size = args.min_worker
        # worker_size = min(math.ceil(args.batch_size / 2.0), args.max_worker)
        logger.info("Scaling down dask cluster to %s", worker_size)
        scale_cluster_and_wait(client, worker_size)

    # NOTE: if data is scattered before scale-down, the error will be thrown:
    # Removing worker 'tcp://<worker IP & port>' caused the cluster to lose scattered data, which can't be recovered
    df, covar_set_id, ranked_features = augment_anchor_df_with_covars(
        anchor_df, args, alchemyEngine, logger, cutoff_date
    )
    # df_future = client.scatter(df)
    # ranked_features_future = client.scatter(ranked_features)

    if args.resume == "predict":
        return hps_id, cutoff_date, ranked_features, df

    min_covar_loss = min_covar_loss_val(
        alchemyEngine, args.model, symbol, args.symbol_table, cutoff_date
    )
    min_covar_loss = min_covar_loss if min_covar_loss is not None else LOSS_CAP
    base_loss = min(base_loss, min_covar_loss)

    # run HP search using Bayeopt and check whether needed HP(s) are found
    logger.info(
        "Starting Bayesian optimization search for hyper-parameters. Loss_val threshold: %s",
        round(base_loss, 5),
    )

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
    wait(futures)

    return hps_id, cutoff_date, ranked_features, df


def update_covar_set_id(alchemyEngine, hps_id, covar_set_id):
    sql = """
        update hps_sessions
        set covar_set_id = :covar_set_id
        where id = :hps_id
    """
    with alchemyEngine.begin() as conn:
        conn.execute(text(sql), {"hps_id": hps_id, "covar_set_id": covar_set_id})


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

    _cleanup_stale_keys()

    space_str = _search_space(
        args.model, model, min(args.max_covars, len(ranked_features)), args.topk_covars
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

    domain_size_base = domain_size

    if args.model == "SOFTS":
        df, _ = impute(df, args.random_seed, client)
    if model is not None and not model.accept_missing_data():
        df, _ = model.impute(df, random_seed=args.random_seed)

    # locks = get_accelerator_locks(cpu_leases=0, timeout="60s")
    # split large iterations into smaller runs to avoid OOM / memory leak
    for i in range(args.max_itr):
        logger.info(
            "running bayesopt mini-iteration %s/%s batch size: %s  domain size: %s runs: %s",
            i + 1,
            args.max_itr,
            n_jobs,
            domain_size,
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
            args.resume.lower() != "none" or i > 0,
        )

        if domain_size:
            domain_size += domain_size_base

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
            # client.restart()
            restart_all_workers(client)
