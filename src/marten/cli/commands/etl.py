from marten.data.etl import main, run_main_with_profiling
from marten.utils.logger import get_logger

def configure_parser(etl_parser):
    etl_parser.add_argument("--profile", action="store_true", help="Profile the ETL process e.g. to troubleshoot bottlenecks")
    # etl_parser.add_argument("--dest", help="Destination to load data into")
    etl_parser.set_defaults(func=handle_etl)


def handle_etl(args):
    logger = get_logger(__name__)

    try:
        if args.profile:
            run_main_with_profiling()
        else:
            main()

        logger.info("ETL process completed successfully.")
    except Exception as e:
        logger.exception("main process terminated")