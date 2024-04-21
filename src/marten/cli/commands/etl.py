from marten.data.etl import main, run_main_with_profiling
from marten.utils.logger import get_logger

def configure_parser(parser):
    parser.add_argument("--profile", action="store_true", help="Profile the ETL process e.g. to troubleshoot bottlenecks")

    parser.add_argument(
        "--worker",
        action="store",
        type=int,
        default=-1,
        help=(
            "Number of parallel workers (python processes) for data retrieval. "
            "Defaults to using all CPU cores available."
        ),
    )
    parser.add_argument(
        "--threads",
        action="store",
        type=int,
        default=3,
        help=(
            "Number of threads per worker (python processes) for data retrieval. "
            "Defaults to 3."
        ),
    )
    parser.add_argument(
        "--dashboard_port",
        action="store",
        type=int,
        default=8788,
        help=(
            "Port number for the dask dashboard. "
            "Defaults to 8788."
        ),
    )

    parser.set_defaults(func=handle_etl)


def handle_etl(args):
    logger = get_logger(__name__)

    try:
        if args.profile:
            run_main_with_profiling(args)
        else:
            main(args)

        logger.info("ETL process completed successfully.")
    except Exception as e:
        logger.exception("main process terminated")
