from marten.models.gridsearch import main
from marten.utils.logger import get_logger


def configure_parser(parser):
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
    group2.add_argument(
        "--max_covars",
        action="store",
        type=int,
        default=100,
        help=(
            "Limit the maximum number of top-covariates to be included for training and prediction. "
            "If it's less than 1, we'll use all covariates with loss_val less than univariate baseline."
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
        help="Epochs for training the model. Defaults to 500",
    )
    parser.add_argument(
        "--worker",
        action="store",
        type=int,
        default=-1,
        help=("Number or parallel workers (python processes) for training the model. "
              "Defaults to using 80% of CPU cores available."
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
        "--infer_holiday", action="store_true", 
        help=("Infer holiday region based on anchor symbol's nature, "
              "which will be utilized during covariate-searching and grid-search.")
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="Use early stopping during model fitting",
    )
    parser.add_argument(
        "--dashboard_port",
        action="store",
        type=int,
        default=8787,
        help=("Port number for the dask dashboard. " "Defaults to 8787."),
    )

    parser.add_argument(
        "symbol", type=str, help="The asset symbol as anchor to be analyzed."
    )

    parser.set_defaults(func=handle_gs)


def handle_gs(args):
    logger = get_logger(__name__)
    try:
        main(args)
        logger.info("grid-search process completed successfully.")
    except Exception:
        logger.exception("grid-search main process terminated")
