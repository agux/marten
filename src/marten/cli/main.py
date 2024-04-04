import argparse
from marten.cli.commands import etl


def main():
    parser = argparse.ArgumentParser(
        description="Financial Market Data Analytics and Prediction CLI"
    )
    subparsers = parser.add_subparsers(
        title="subcommands", description="valid subcommands", help="additional help"
    )

    # ETL Sub-command
    etl_parser = subparsers.add_parser("etl", help="Extract, transform, and load data")
    etl.configure_parser(etl_parser)

    # # Training Sub-command
    # train_parser = subparsers.add_parser("train", help="Train models on financial data")
    # train.configure_parser(train_parser)

    # # Hyperparameter Tuning Sub-command
    # tune_parser = subparsers.add_parser("tune", help="Tune hyperparameters for models")
    # tune.configure_parser(tune_parser)

    # # Multivariate Searching Sub-command
    # search_parser = subparsers.add_parser(
    #     "search", help="Perform multivariate searching"
    # )
    # search.configure_parser(search_parser)

    # # Analytics Sub-command
    # analyze_parser = subparsers.add_parser(
    #     "analyze", help="Run analytics on financial data"
    # )
    # analyze.configure_parser(analyze_parser)

    # # Portfolio Design Sub-command
    # portfolio_parser = subparsers.add_parser(
    #     "portfolio", help="Design and analyze portfolios"
    # )
    # portfolio.configure_parser(portfolio_parser)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
