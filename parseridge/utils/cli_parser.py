import argparse


def parse_cli_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--train",
        type=str,
        help="Path to train file.",
        required=True
    )

    parser.add_argument(
        "--test",
        type=str,
        help="Path to test file.",
        required=True
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of sentences per batch.",
        required=False
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123456789,
        help="Number to initialize randomness with.",
        required=False
    )

    parser.add_argument(
        "--error_probability",
        type=float,
        default=0.1,
        help="Probability to make a bad decision during parsing to increase"
             "robustness.",
        required=False
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.33,
        help="Dropout used.",
        required=False
    )
    return parser.parse_args()
