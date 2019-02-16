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
        help="Number of sentences per batch.",
        required=True
    )


    return parser.parse_args()
