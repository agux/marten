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
            "Number or parallel workers (python processes) for data retrieval. "
            "Defaults to using all CPU cores available."
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
