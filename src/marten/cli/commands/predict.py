from marten.models.predict import main
from marten.utils.logger import get_logger


def configure_parser(parser):
    parser.add_argument(
        "--epochs",
        action="store",
        type=int,
        default=500,
        help="Maximum epochs for model fitting. Defaults to 500",
    )
    parser.add_argument(
        "--worker",
        action="store",
        type=int,
        default=-1,
        help=("Number or parallel workers (python processes) for model fitting and prediction. "
              "Defaults to the number of symbols provided, or the number of cpu cores, whichever is smaller."
        )
    )
    parser.add_argument(
        "--timestep_limit",
        action="store",
        type=int,
        default=1200,
        help=(
            "Limit the time steps of anchor symbol to the most recent N data points. "
            "Specify -1 to utilize all time steps available. "
            "Defaults to 1200"
        ),
    )
    parser.add_argument(
        "--random_seed",
        action="store",
        type=int,
        default=7,
        help=(
            "Random seed for the stochastic model fitting / prediction process."
            "Defaults to 7."
        ),
    )
    parser.add_argument(
        "--future_steps", action="store", type=int, default=60, help="Specify how many time steps (days) into the future will be predicted. Defaults to 60."
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="Use early stopping during model fitting",
    )

    parser.add_argument(
        "symbols", metavar='S', type=str, nargs='+', help="Array of asset symbols to be predicted."
    )

    parser.set_defaults(func=handle_predict)


def handle_predict(args):
    logger = get_logger(__name__)
    try:
        main(args)
        logger.info("Prediction for %s completed successfully.", args.symbols)
    except Exception:
        logger.exception("Prediction main process terminated")
