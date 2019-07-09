import argparse

from parseridge.parser.loss import Criterion


def parse_cli_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    file_group = parser.add_argument_group("Files")
    file_group.add_argument(
        "--train_corpus", type=str, help="Path to train file.", required=True
    )

    file_group.add_argument(
        "--test_corpus", type=str, help="Path to test file.", required=True
    )

    # TODO add files to save and load models
    # TODO add files to save treebank output

    nn_group = parser.add_argument_group("Model Design")
    nn_group.add_argument(
        "--num_stack",
        type=int,
        default=3,
        help="Number of top items on the stack to use for classification.",
        required=False,
    )

    nn_group.add_argument(
        "--num_buffer",
        type=int,
        default=1,
        help="Number of top items on the buffer to use for classification.",
        required=False,
    )

    nn_group.add_argument(
        "--embedding_size",
        type=int,
        default=100,
        help="Size of the word embeddings. "
        "Will be ignored, if external embeddings are loaded.",
        required=False,
    )

    nn_group.add_argument(
        "--lstm_hidden_size",
        type=int,
        default=125,
        help="Size of the hidden layer of the LSTM. The output size will be 2x the size.",
        required=False,
    )

    nn_group.add_argument(
        "--lstm_layers", type=int, default=2, help="Number of stacked LSTMs", required=False
    )

    nn_group.add_argument(
        "--relation_mlp_layers",
        type=int,
        default=[100],
        nargs="+",
        help="List of sizes of the layers in the MLP for the relation labels.",
        required=False,
    )

    nn_group.add_argument(
        "--transition_mlp_layers",
        type=int,
        default=[100],
        nargs="+",
        help="List of sizes of the layers in the MLP for the transitions.",
        required=False,
    )

    # TODO add activation function here?
    regularization_group = parser.add_argument_group("Regularization")
    regularization_group.add_argument(
        "--margin_threshold",
        type=float,
        default=2.5,
        help="The desired difference between the best right and the best wrong action.",
        required=False,
    )

    regularization_group.add_argument(
        "--error_probability",
        type=float,
        default=0.1,
        help="The probability to induce an error by choosing a wrong action.",
        required=False,
    )

    regularization_group.add_argument(
        "--oov_probability",
        type=float,
        default=0.25,
        help="A percentage to randomly replace tokens by the OOV vector: "
        "freq / (freq + oov_prob).",
        required=False,
    )

    regularization_group.add_argument(
        "--update_size",
        type=int,
        default=50,
        help="Update the weights after accumulating a certain number of losses.",
        required=False,
    )

    regularization_group.add_argument(
        "--loss_factor",
        type=float,
        default=0.75,
        help="Multiply the accumulated loss with this number to regularize it.",
        required=False,
    )

    regularization_group.add_argument(
        "--loss_strategy",
        type=str,
        default="avg",
        help="Strategy to reduce a list of loss values to one. Supported are avg and sum.",
        required=False,
    )

    regularization_group.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="The learning rate for the Adam trainer.",
        required=False,
    )

    regularization_group.add_argument(
        "--weight_decay",
        type=float,
        default=0.00,
        help="Regularize the weights during an update.",
        required=False,
    )

    regularization_group.add_argument(
        "--gradient_clipping",
        type=float,
        default=10.0,
        help="Make sure gradients do not get larger than this.",
        required=False,
    )

    regularization_group.add_argument(
        "--token_dropout",
        type=float,
        default=0.01,
        help="Probability that any token will be replaced by an OOV token.",
        required=False,
    )

    regularization_group.add_argument(
        "--lstm_dropout",
        type=float,
        default=0.33,
        help="Dropout used between the stacked LSTMs. Note that there is no support for "
        "recurrent dropout.",
        required=False,
    )

    regularization_group.add_argument(
        "--mlp_dropout",
        type=float,
        default=0.25,
        help="Dropout used between layers in the MLPs.",
        required=False,
    )

    regularization_group.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of sentences per batch.",
        required=False,
    )

    regularization_group.add_argument(
        "--loss_function",
        type=str,
        default="CrossEntropy",
        help="Name of the loss function to use.",
        required=False,
        choices=list(Criterion.LOSS_FUNCTIONS.keys()),
    )

    analytics_group = parser.add_argument_group("Analytics")
    analytics_group.add_argument(
        "--google_sheet_id",
        type=str,
        help="The id of the Google Sheet to save the report in.",
        required=False,
    )

    analytics_group.add_argument(
        "--google_sheet_auth_file",
        type=str,
        help="The auth.json file to for the Google API.",
        required=False,
        default="google_sheets_auth.json",
    )

    parser.add_argument(
        "--embeddings_file",
        type=str,
        default="",
        help="Path to external embeddings to load.",
        required=False,
    )

    parser.add_argument(
        "--comment",
        type=str,
        default="",
        help="A comment about this experiment.",
        required=False,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Number to initialize randomness with.",
        required=False,
    )

    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of epochs to run.", required=False
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on. cpu or cuda.",
        required=False,
    )

    parser.add_argument(
        "--train",
        action="store_true",
        default=True,
        help="Use in training mode.",
        required=False,
    )

    parser.add_argument(
        "--pred_batch_size",
        type=int,
        default=512,
        help="Predict number of sentences per batch.",
        required=False,
    )

    return parser.parse_args()
