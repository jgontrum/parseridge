import argparse

from distutils.util import strtobool

from parseridge.parser.activation import ACTIVATION_FUNCTIONS
from parseridge.parser.modules.attention.soft_attention import Attention
from parseridge.parser.modules.configuration_encoder import CONFIGURATION_ENCODERS
from parseridge.parser.modules.input_encoder import InputEncoder


def parse_train_cli_arguments():
    parser = argparse.ArgumentParser(
        prog="python parseridge/train.py",
        description="Trains a parser model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    file_group = parser.add_argument_group("File Paths")
    file_group.add_argument(
        "--model_save_path",
        type=str,
        help="If set, the models are saved in this directory after each epoch.",
        required=False,
    )

    file_group.add_argument(
        "--conllu_save_path",
        type=str,
        help="If set, parsed all parsed sentences will be saved in this directory.",
        required=False,
    )

    file_group.add_argument(
        "--csv_output_file",
        type=str,
        help="If set, the results are saved in this csv file.",
        required=False,
    )

    file_group.add_argument(
        "--yml_output_file",
        type=str,
        help="If set, the results are saved in this yml file.",
        required=False,
    )

    file_group.add_argument(
        "--log_file",
        type=str,
        help="If set, the log is saved in this file.",
        required=False,
    )

    file_group.add_argument(
        "--embeddings_file",
        type=str,
        default="",
        help="Path to external embeddings to load.",
        required=False,
    )

    file_group.add_argument(
        "--train_corpus", type=str, help="Path to the train.conllu file.", required=True
    )

    file_group.add_argument(
        "--dev_corpus", type=str, help="Path to the dev.conllu file.", required=True
    )

    file_group.add_argument(
        "--test_corpus", type=str, help="Path to the test.conllu file.", required=False
    )

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
        "--input_encoder_type",
        type=str,
        default="lstm",
        help="The type of input encoder to use.",
        required=False,
        choices=InputEncoder.INPUT_ENCODER_MODES,
    )

    nn_group.add_argument(
        "--lstm_hidden_size",
        type=int,
        default=125,
        help="Size of the hidden layer of the LSTM. The output size will be 2x the size.",
        required=False,
    )

    nn_group.add_argument(
        "--lstm_layers", type=int, default=3, help="Number of stacked LSTMs", required=False
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

    nn_group.add_argument(
        "--relation_mlp_activation",
        type=str,
        default="tanh",
        help="Activation function for the relation MLP.",
        required=False,
        choices=list(ACTIVATION_FUNCTIONS.keys()),
    )

    nn_group.add_argument(
        "--transition_mlp_activation",
        type=str,
        default="tanh",
        help="Activation function for the transition MLP.",
        required=False,
        choices=list(ACTIVATION_FUNCTIONS.keys()),
    )

    nn_group.add_argument(
        "--mlp_input_transformation_layers",
        type=int,
        default=[],
        nargs="*",
        help="List of sizes of the layers used to transform the input to the MLPs.",
        required=False,
    )

    nn_group.add_argument(
        "--encoder_output_transformation_layers",
        type=int,
        default=[],
        nargs="*",
        help="List of sizes of the layers used to transform the output of the encoder.",
        required=False,
    )

    regularization_group = parser.add_argument_group("Regularization")
    regularization_group.add_argument(
        "--margin_threshold",
        type=float,
        default=1.0,
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
        "--update_frequency",
        type=int,
        default=50,
        help="Update the weights after accumulating a certain number of losses.",
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
        default=100.0,
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
        default="MaxMargin",
        help="Name of the loss function to use.",
        required=False,
        choices=["MaxMargin", "CrossEntropy"],
    )

    regularization_group.add_argument(
        "--dimensionality_reduction",
        type=int,
        default=0,
        help="Transform the output of the input encoder into this dimension.",
        required=False,
    )

    attention_group = parser.add_argument_group("Attention")
    attention_group.add_argument(
        "--configuration_encoder",
        type=str,
        default="static",
        help="The method how to represent the current configuration as input to the MLP.",
        required=False,
        choices=list(CONFIGURATION_ENCODERS.keys()),
    )

    attention_group.add_argument(
        "--attention_reporter_path",
        type=str,
        help="Path to a folder where all the attention weights are logged to.",
        required=False,
    )

    attention_group.add_argument(
        "--scale_query",
        type=int,
        default=None,
        help="If set, scale the query vectors to this dimension.",
        required=False,
    )

    attention_group.add_argument(
        "--scale_key",
        type=int,
        default=None,
        help="If set, scale the key vectors to this dimension.",
        required=False,
    )

    attention_group.add_argument(
        "--scale_value",
        type=int,
        default=None,
        help="If set, scale the value vectors to this dimension. "
        "Must be equal to 'scale_key'.",
        required=False,
    )

    attention_group.add_argument(
        "--scoring_function",
        type=str,
        default="concat",
        help="Name of the scoring function to use.",
        required=False,
        choices=list(Attention.SCORING_FUNCTIONS.keys()),
    )

    attention_group.add_argument(
        "--normalization_function",
        type=str,
        default="softmax",
        help="Name of the normalization function to use.",
        required=False,
        choices=list(Attention.NORMALIZATION_FUNCTIONS.keys()),
    )

    attention_group.add_argument(
        "--self_attention_heads",
        type=int,
        default=10,
        help="Number of heads in the self-attention encoder if used. "
        "The encoding dimensions must be dividable by this number.",
        required=False,
    )

    attention_group.add_argument(
        "--self_attention_layers",
        type=int,
        default=2,
        help="Stacked self-attention layers.",
        required=False,
    )

    misc_group = parser.add_argument_group("Misc.")
    misc_group.add_argument(
        "--google_sheets_id",
        type=str,
        help="The id of the Google Sheet to save the report in.",
        required=False,
    )

    misc_group.add_argument(
        "--google_sheets_auth_path",
        type=str,
        help="The auth.json file to for the Google API.",
        required=False,
    )

    misc_group.add_argument(
        "--embeddings_vendor",
        type=str,
        default="glove",
        help="Name of the embeddings format.",
        required=False,
        choices=["glove", "fasttext"],
    )

    misc_group.add_argument(
        "--freeze_embeddings",
        default=True,
        type=lambda x: bool(strtobool(x)),
        help="Freeze the external embeddngs or not.",
        required=False,
    )

    misc_group.add_argument(
        "--show_progress_bars",
        default=True,
        type=lambda x: bool(strtobool(x)),
        help="Show the progress bars for training and evaluation or not.",
        required=False,
    )

    misc_group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Number to initialize randomness with.",
        required=False,
    )

    misc_group.add_argument(
        "--experiment_name",
        type=str,
        default="",
        help="Name of the experiment. Used for e.g. for logging.",
        required=False,
    )

    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs to run.", required=False
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on. cpu or cuda.",
        required=False,
    )

    parser.add_argument(
        "--commit",
        type=str,
        default=None,
        help="Optional git commit this experiment is supposed to run at.",
        required=False,
    )

    return parser.parse_args()
