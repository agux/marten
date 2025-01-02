from marten.models.predict import main
from marten.utils.logger import get_logger


def configure_parser(parser):
    parser.add_argument(
        "--model",
        choices=["NeuralProphet", "SOFTS", "TimeMixer", "TSMixerx"],
        default="TSMixerx",
        help="Model to be used for the prediction. Defaults to TSMixerx.",
    )
    parser.add_argument(
        "--epochs",
        action="store",
        type=int,
        default=5000,
        help="Maximum epochs for model fitting. Defaults to 5000",
    )
    parser.add_argument(
        "--min_worker",
        action="store",
        type=int,
        default=4,
        help=(
            "Specifies minimum parallel workers (python processes) to keep. "
            "Defaults to 4."
        ),
    )
    parser.add_argument(
        "--max_worker",
        action="store",
        type=int,
        default=-1,
        help=(
            "Specifies maximum parallel workers (python processes). "
            "Defaults to the number of cpu cores."
        ),
    )
    parser.add_argument(
        "--threads",
        action="store",
        type=int,
        default=1,
        help=(
            "Number of threads per worker (python processes)"
            "Defaults to 1"
        ),
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
        "--max_covars",
        action="store",
        type=int,
        default=1000,
        help=(
            "Limit the maximum number of top-covariates to be included for training and prediction. "
            "If it's less than 1, we'll use all covariates with loss_val less than univariate baseline. "
            "Defaults to 1000."
        ),
    )
    parser.add_argument(
        "--topk_covars",
        action="store",
        type=int,
        default=50,
        help=(
            "Select the top-k covariates from the 'max_covars' for training and prediction. "
            "If it's less than 1, we'll use up to all 'max_covar'."
            "Defaults to 50."
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
        "--future_steps",
        action="store",
        type=int,
        default=20,
        help="Specify how many time steps (days) into the future will be predicted. Defaults to 20.",
    )
    parser.add_argument(
        "--topk",
        action="store",
        type=int,
        default=3,
        help="Use top-k best historical hyper-parameters to ensemble the prediction. Default is 3.",
    )
    parser.add_argument(
        "--gpu_util_threshold",
        action="store",
        type=int,
        default=85,
        help=(
            "When accelerator is switched on, "
            "it will fall back to CPU when GPU processor utilization is over the given percentage. "
            "Default is 85."
        ),
    )
    parser.add_argument(
        "--gpu_ram_threshold",
        action="store",
        type=int,
        default=85,
        help=(
            "When accelerator is switched on, "
            "it will fall back to CPU when GPU vRAM utilization is over the given percentage. "
            "Default is 85."
        ),
    )
    parser.add_argument(
        "--wait_gpu",
        action="store",
        type=int,
        default=600,
        help=(
            "When the number of covariates exceed the given value, "
            "the task will try to wait for the availability of GPU before falling back to CPU. "
            "Defaults to 600."
        ),
    )
    parser.add_argument(
        "--batch_size",
        action="store",
        type=int,
        default=200,
        help=(
            "Batch size for each iteration of the Bayesian optimized search. "
            "Defaults to 200"
        ),
    )
    parser.add_argument(
        "--mini_itr",
        action="store",
        type=int,
        default=5,
        help=(
            "Mini-iteration for an inner Bayes Optimization run "
            "until `topk` HP of which Loss_val is less than baseline. "
            "Defaults to 5"
        ),
    )
    parser.add_argument(
        "--max_itr",
        action="store",
        type=int,
        default=10,
        help=(
            "Max iteration for the outer Bayes Optimization. "
            "Total HP searched would be (batch_size * mini_itr * max_itr). "
            "Defaults to 10"
        ),
    )
    parser.add_argument(
        "--domain_size",
        action="store",
        type=int,
        default=10000,
        help=(
            "Limit domain size for the Bayesian Optimization HP search space."
            "Larger size demands more memory for search iteration. "
            "Defaults to 10000"
        ),
    )
    parser.add_argument(
        "--resource_intensive_sql_semaphore",
        action="store",
        type=int,
        default=8,
        help=(
            "Semaphore to limit resource intensive sql execution in parallel."
            "Defaults to 8"
        ),
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="Use early stopping during model fitting",
    )
    parser.add_argument(
        "--dask_log",
        action="store_true",
        help="Switch log level to INFO for dask components. Default is WARN.",
    )
    parser.add_argument(
        "--restart_workers",
        action="store_true",
        help="Restart all workers in the dask cluster at each end of mini-iteration.",
    )
    parser.add_argument(
        "--log_train_args",
        action="store_true",
        help="Logs model training arguments at info level.",
    )
    parser.add_argument(
        "--resume",
        choices=["none", "covar", "hps"],
        default="none",
        help=(
            "Continue from the specified prediction process. "
            "If not specified, we'll start from scratch with latest time-series data, "
            "which is the default behavior."
        ),
    )
    parser.add_argument(
        "--adhoc",
        action="store_true",
        help=(
            "Perform adhoc prediction. "
            "Re-calculate the covariate validation loss and perform HP search using latest historical data. "
            "The search will be stopped at least`topk` HP has been found for ensemble predictions."
        ),
    )
    parser.add_argument(
        "--loss_quantile",
        action="store",
        type=float,
        default=0.85,
        help=(
            "During Bayesian HP search, only those with loss_val below `baseline_loss_val * loss_quantile` "
            "will be considered qualified HP for ensemble prediction. "
            "Defaults to 0.9"
        ),
    )
    parser.add_argument(
        "--nan_limit",
        action="store",
        type=float,
        default=0.005,
        help=(
            "Limit the ratio of NaN (missing data) in covariates. "
            "Only those with NaN rate lower than the limit ratio can be selected during multivariate HP searching. "
            "Defaults to 0.005."
        ),
    )
    parser.add_argument(
        "--accelerator", action="store_true", help="Use accelerator automatically"
    )
    parser.add_argument(
        "--dashboard_port",
        action="store",
        type=int,
        default=8789,
        help=("Port number for the dask dashboard. " "Defaults to 8789."),
    )
    parser.add_argument(
        "--scheduler_port",
        action="store",
        type=int,
        default=0,
        help=("Port number for the scheduler. " "Defaults to 0 (a random port will be selected)"),
    )
    parser.add_argument(
        "--asset_type",
        choices=["stock", "index", "ETF"],
        default="ETF",
        help="Asset type of the symbol(s). Defaults to ETF.",
    )
    parser.add_argument(
        "symbols", type=str, nargs="+", help="Array of asset symbols to be predicted."
    )

    parser.set_defaults(func=handle_predict)


def handle_predict(args):
    logger = get_logger(__name__)
    try:
        main(args)
        logger.info("Prediction for %s completed successfully.", args.symbols)
    except Exception:
        logger.exception("Prediction main process terminated")
