import math
import uuid
import pandas as pd

from typing import List, Any

from sqlalchemy import Engine

from dask.distributed import get_worker, worker_client, Client, Future, wait

from tsfresh import extract_relevant_features
from tsfresh.utilities.dataframe_functions import roll_time_series

from marten.utils.logger import get_logger
from marten.utils.database import tables_with_prefix
from marten.utils.worker import get_results
from marten.data.db import update_on_conflict
from marten.data.tabledef import table_def_ts_features
from marten.models.worker_func import fit_with_covar


def extract_features(
    client: Client,
    alchemyEngine: Engine,
    symbol: str,
    anchor_df: pd.DataFrame,
    anchor_table: str,
    args: Any,
) -> List[Future]:
    ts_date = anchor_df["ds"].max().strftime("%Y-%m-%d")
    # val_size = validation_size(anchor_df)
    # anchor_df = anchor_df.iloc[:-val_size, :].copy()
    # extract features from endogenous variables and all features of top-N assets
    targets = anchor_df[["ds", "y"]]
    targets.loc[:, "target"] = targets["y"].shift(-1).apply(lambda x: 1 if x > 0 else 0)
    targets = targets.dropna(subset=["target"])
    targets = targets[["ds", "target"]]
    # 1. extract features from endogenous variables
    futures = []
    futures.append(
        client.submit(
            extract_features_on,
            symbol,
            anchor_table,
            symbol,
            anchor_table,
            anchor_df,
            anchor_df,
            targets,
        )
    )
    get_logger().info("getting top %s covariates...", args.max_covars)
    # 2. select top-N symbols from paired_correlation
    query = """
        WITH cte0 AS (
            SELECT
                cov_table, cov_symbol
            FROM
                paired_correlation pc
            WHERE
                pc.model = %(model)s
                AND pc.symbol = %(anchor_symbol)s
                AND pc.symbol_table = %(symbol_table)s
                AND pc.ts_date = %(ts_date)s
            ORDER BY loss_val
            LIMIT %(limit)s
        ), cte AS (
            SELECT * FROM cte0 WHERE cov_table NOT LIKE 'ta|_%%' ESCAPE '|'
            UNION ALL
            SELECT 
                (string_to_array(cov_symbol, '::'))[1] AS cov_table,
                (string_to_array(cov_symbol, '::'))[2] AS cov_symbol
            FROM cte0 WHERE cov_table LIKE 'ta|_%%' ESCAPE '|'
        )
        SELECT DISTINCT ON (cov_table, cov_symbol) * FROM cte
    """

    params = {
        "model": args.model,
        "anchor_symbol": symbol,
        "symbol_table": anchor_table,
        "ts_date": ts_date,
        "limit": args.max_covars,
    }
    with alchemyEngine.connect() as conn:
        topk_covars = pd.read_sql(
            query,
            conn,
            params=params,
        )
    # query a list of ta_ table names from meta table
    ta_tables = tables_with_prefix(alchemyEngine, "ta")

    from marten.models.hp_search import load_anchor_ts

    n_jobs = int(math.sqrt(args.min_worker * args.max_worker))
    for cov_table, cov_symbol in topk_covars.itertuples(index=False):
        # for each symbol, extract features from basic table
        feature_df, _ = load_anchor_ts(cov_symbol, 0, alchemyEngine, ts_date, cov_table)
        cov_table = cov_table[:-5] if cov_table.endswith("_view") else cov_table
        futures.append(
            client.submit(
                extract_features_on,
                symbol,
                anchor_table,
                cov_symbol,
                cov_table,
                anchor_df,
                feature_df,
                targets,
            )
        )

        # extract features from TA table
        for ta_table in ta_tables:
            ta_view = ta_table + "_view"
            with alchemyEngine.connect() as conn:
                ta_df = pd.read_sql(
                    f"""
                        select * from {ta_view} 
                        where "table" = %(table)s
                        and symbol = %(symbol)s
                    """,
                    con=conn,
                    params={"table": cov_table, "symbol": cov_symbol},
                    parse_dates=["date"],
                )

                ta_df = ta_df.drop(columns=["table", "symbol", "last_modified"]).rename(
                    columns={"date": "ds"}
                )

            futures.append(
                client.submit(
                    extract_features_on,
                    symbol,
                    anchor_table,
                    f"{cov_table}::{cov_symbol}",
                    ta_view,
                    anchor_df,
                    ta_df,
                    targets,
                )
            )

            if len(futures) > n_jobs:
                done, undone = wait(futures, return_when="FIRST_COMPLETED")
                get_results(done)
                del done
                futures = list(undone)

    return futures


def _save_result(
    symbol: str,
    symbol_table: str,
    cov_symbol: str,
    cov_table: str,
    date: str,
    num_features: int,
) -> None:
    """
    Persist a single row to the `ts_features_result` table, upserting on conflict.

    :param symbol: Main symbol (endogenous variable).
    :param symbol_table: Table name for the main symbol.
    :param cov_symbol: Covariate symbol.
    :param cov_table: Table name for the covariate.
    :param date: Date string in YYYY-MM-DD format.
    :param num_features: The number of extracted features.
    """
    worker = get_worker()
    alchemyEngine = worker.alchemyEngine

    statement = """
        INSERT INTO public.ts_features_result
        (symbol_table, symbol, cov_table, cov_symbol, "date", num_features)
        VALUES (:symbol_table, :symbol, :cov_table, :cov_symbol, :date, :num_features)
        ON CONFLICT (symbol_table, symbol, cov_table, cov_symbol, "date")
        DO UPDATE
            SET num_features = EXCLUDED.num_features
    """

    params = {
        "symbol_table": symbol_table,
        "symbol": symbol,
        "cov_table": cov_table,
        "cov_symbol": cov_symbol,
        "date": date,
        "num_features": num_features,
    }

    with alchemyEngine.begin() as conn:
        conn.execute(statement, params)


def _get_result(
    symbol: str, symbol_table: str, cov_symbol: str, cov_table: str, date: str
) -> int:
    worker = get_worker()
    alchemyEngine = worker.alchemyEngine
    query = """
        SELECT num_features
        FROM public.ts_features_result
        WHERE symbol_table = %(symbol_table)s
          AND symbol = %(symbol)s
          AND cov_table = %(cov_table)s
          AND cov_symbol = %(cov_symbol)s
          AND "date" = %(date)s
    """
    params = {
        "symbol_table": symbol_table,
        "symbol": symbol,
        "cov_table": cov_table,
        "cov_symbol": cov_symbol,
        "date": date,
    }

    with alchemyEngine.connect() as conn:
        df = pd.read_sql(query, conn, params=params)
        # Return the integer value if a record is found; otherwise None
        return df["num_features"].iloc[0] if not df.empty else None


def extract_features_on(
    symbol: str,
    symbol_table: str,
    cov_symbol: str,
    cov_table: str,
    anchor_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    targets: pd.DataFrame,
):
    worker = get_worker()
    args = worker.args
    alchemyEngine = worker.alchemyEngine
    logger = worker.logger
    futures = []

    # check if the record already exists in ts_features, and skip processing
    max_date = targets["ds"].max().strftime("%Y-%m-%d")
    num_features = _get_result(
        symbol,
        symbol_table,
        cov_symbol,
        cov_table,
        max_date,
    )
    if num_features is not None:
        logger.info(
            "%s features have been processed for %s@%s previously, skipping",
            num_features,
            cov_symbol,
            cov_table,
        )
        return num_features

    logger.info(
        "extracting features for covar %s@%s",
        cov_symbol,
        cov_table,
    )
    na_counts = feature_df.isna().sum()
    thres = len(feature_df) * 0.2
    cols_to_drop = []
    for c in feature_df.columns:
        if na_counts[c] > thres:
            cols_to_drop.append(c)
    df = feature_df.drop(columns=cols_to_drop).dropna(how="any")
    # df = feature_df.dropna(how="any")
    df.insert(0, "unique_id", symbol)
    rts = roll_time_series(
        df,
        column_id="unique_id",
        column_sort="ds",
        min_timeshift=5,
        max_timeshift=20,
        n_jobs=0,
        disable_progressbar=True,
    )
    rts = rts.merge(targets, on="ds", how="left")
    rts = rts.dropna(subset=["target"])

    y = rts.groupby("id")["target"].last()
    x = rts.drop(columns=["unique_id", "target"])

    logger.debug("x:\n %s", x)
    logger.debug("y:\n %s", y)
    # now = datetime.now().strftime("%Y%m%d%H%M%S")
    # x.to_pickle(f"x_{now}.pkl")
    # y.to_pickle(f"y_{now}.pkl")

    # logger.info("client address: %s", client.cluster.scheduler_address)
    # distributor = ClusterDaskDistributor(address=client.cluster.scheduler_address)

    try:
        features = extract_relevant_features(
            x,
            y,
            column_id="id",
            column_sort="ds",
            n_jobs=0,
            disable_progressbar=True,
        )
    except Exception as e:
        logger.error(e, exc_info=True)
        raise e

    if features.empty:
        logger.info(
            "empty features extracted for covar %s@%s",
            cov_symbol,
            cov_table,
        )
        _save_result(symbol, symbol_table, cov_symbol, cov_table, max_date, 0)
        return 0

    num_features = len(features.columns)
    logger.info(
        "extracted %s relevant features for covar %s@%s",
        num_features,
        cov_symbol,
        cov_table,
    )

    features = features.reset_index().rename(
        columns={"level_0": "symbol", "level_1": "date"}
    )
    eav_df = features.melt(
        id_vars=["symbol", "date"], var_name="feature", value_name="value"
    )
    eav_df.insert(0, "symbol_table", symbol_table)
    eav_df.insert(2, "cov_table", cov_table)
    eav_df.insert(3, "cov_symbol", cov_symbol)
    # save these extracted features to an Entity-Attribute-Value table.
    with alchemyEngine.begin() as conn:
        update_on_conflict(
            table_def_ts_features(),
            conn,
            eav_df,
            ["symbol_table", "symbol", "cov_table", "cov_symbol", "feature", "date"],
        )
        # eav_df.to_sql("ts_features", con=conn, if_exists="append", index=False)

    features = features.rename(columns={"date": "ds"})

    min_date = anchor_df["ds"].min().strftime("%Y-%m-%d")
    feat_cols = [c for c in features.columns if c not in ("symbol", "ds")]

    # re-run paired correlation on these features in parallel
    with worker_client() as client:
        for fcol in feat_cols:
            df = anchor_df[["ds", "y"]].merge(
                features[["ds", fcol]], on="ds", how="left"
            )
            futures.append(
                client.submit(
                    fit_with_covar,
                    symbol,
                    df,
                    "ts_features_view",
                    f"{cov_table}::{cov_symbol}",
                    min_date,
                    args.random_seed,
                    fcol,
                    "auto",
                    args.early_stopping,
                    True,
                    key=f"{fit_with_covar.__name__}({cov_symbol})-{uuid.uuid4().hex}",
                )
            )
            if len(futures) > args.max_worker:
                done, undone = wait(futures, return_when="FIRST_COMPLETED")
                get_results(done)
                del done
                futures = list(undone)

        done, _ = wait(futures)
        get_results(done)
        del done
        _save_result(symbol, symbol_table, cov_symbol, cov_table, max_date, num_features)

    return num_features
