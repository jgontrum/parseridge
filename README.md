# Parseridge
A Transition-based Dependency Parser in PyTorch.

## Usage
```bash
usage: python parseridge/train.py [-h] [--model_save_path MODEL_SAVE_PATH]
                                  [--csv_output_file CSV_OUTPUT_FILE]
                                  [--log_file LOG_FILE]
                                  [--embeddings_file EMBEDDINGS_FILE]
                                  --train_corpus TRAIN_CORPUS --dev_corpus
                                  DEV_CORPUS [--test_corpus TEST_CORPUS]
                                  [--num_stack NUM_STACK]
                                  [--num_buffer NUM_BUFFER]
                                  [--embedding_size EMBEDDING_SIZE]
                                  [--input_encoder_type {lstm,transformer}]
                                  [--lstm_hidden_size LSTM_HIDDEN_SIZE]
                                  [--lstm_layers LSTM_LAYERS]
                                  [--relation_mlp_layers RELATION_MLP_LAYERS [RELATION_MLP_LAYERS ...]]
                                  [--transition_mlp_layers TRANSITION_MLP_LAYERS [TRANSITION_MLP_LAYERS ...]]
                                  [--relation_mlp_activation {sigmoid,tanh,hard_tanh,relu,leaky_relu,prelu,elu,gelu}]
                                  [--transition_mlp_activation {sigmoid,tanh,hard_tanh,relu,leaky_relu,prelu,elu,gelu}]
                                  [--margin_threshold MARGIN_THRESHOLD]
                                  [--error_probability ERROR_PROBABILITY]
                                  [--oov_probability OOV_PROBABILITY]
                                  [--update_frequency UPDATE_FREQUENCY]
                                  [--learning_rate LEARNING_RATE]
                                  [--weight_decay WEIGHT_DECAY]
                                  [--gradient_clipping GRADIENT_CLIPPING]
                                  [--token_dropout TOKEN_DROPOUT]
                                  [--lstm_dropout LSTM_DROPOUT]
                                  [--mlp_dropout MLP_DROPOUT]
                                  [--batch_size BATCH_SIZE]
                                  [--loss_function {MaxMargin,CrossEntropy}]
                                  [--configuration_encoder {static,universal_attention,stack-buffer_query_attention,finished_tokens_attention,sentence_query_attention}]
                                  [--attention_reporter_path ATTENTION_REPORTER_PATH]
                                  [--scale_query SCALE_QUERY]
                                  [--scale_key SCALE_KEY]
                                  [--scale_value SCALE_VALUE]
                                  [--scoring_function {dot,scaled_dot,general,concat,learned,biaffine,dummy}]
                                  [--normalization_function {softmax,sigmoid,identity}]
                                  [--self_attention_heads SELF_ATTENTION_HEADS]
                                  [--google_sheets_id GOOGLE_SHEETS_ID]
                                  [--google_sheets_auth_path GOOGLE_SHEETS_AUTH_PATH]
                                  [--embeddings_vendor {glove,fasttext}]
                                  [--freeze_embeddings FREEZE_EMBEDDINGS]
                                  [--show_progress_bars SHOW_PROGRESS_BARS]
                                  [--seed SEED]
                                  [--experiment_name EXPERIMENT_NAME]
                                  [--epochs EPOCHS] [--device DEVICE]
                                  [--commit COMMIT]

Trains a parser model.

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       Number of epochs to run. (default: 50)
  --device DEVICE       Device to run on. cpu or cuda. (default: cpu)
  --commit COMMIT       Optional git commit this experiment is supposed to run
                        at. (default: None)

File Paths:
  --model_save_path MODEL_SAVE_PATH
                        If set, the models are saved in this directory after
                        each epoch. (default: None)
  --csv_output_file CSV_OUTPUT_FILE
                        If set, the results are saved in this csv file.
                        (default: None)
  --log_file LOG_FILE   If set, the log is saved in this file. (default: None)
  --embeddings_file EMBEDDINGS_FILE
                        Path to external embeddings to load. (default: )
  --train_corpus TRAIN_CORPUS
                        Path to the train.conllu file. (default: None)
  --dev_corpus DEV_CORPUS
                        Path to the dev.conllu file. (default: None)
  --test_corpus TEST_CORPUS
                        Path to the test.conllu file. (default: None)

Model Design:
  --num_stack NUM_STACK
                        Number of top items on the stack to use for
                        classification. (default: 3)
  --num_buffer NUM_BUFFER
                        Number of top items on the buffer to use for
                        classification. (default: 1)
  --embedding_size EMBEDDING_SIZE
                        Size of the word embeddings. Will be ignored, if
                        external embeddings are loaded. (default: 100)
  --input_encoder_type {lstm,transformer}
                        The type of input encoder to use. (default: lstm)
  --lstm_hidden_size LSTM_HIDDEN_SIZE
                        Size of the hidden layer of the LSTM. The output size
                        will be 2x the size. (default: 125)
  --lstm_layers LSTM_LAYERS
                        Number of stacked LSTMs (default: 3)
  --relation_mlp_layers RELATION_MLP_LAYERS [RELATION_MLP_LAYERS ...]
                        List of sizes of the layers in the MLP for the
                        relation labels. (default: [100])
  --transition_mlp_layers TRANSITION_MLP_LAYERS [TRANSITION_MLP_LAYERS ...]
                        List of sizes of the layers in the MLP for the
                        transitions. (default: [100])
  --relation_mlp_activation {sigmoid,tanh,hard_tanh,relu,leaky_relu,prelu,elu,gelu}
                        Activation function for the relation MLP. (default:
                        tanh)
  --transition_mlp_activation {sigmoid,tanh,hard_tanh,relu,leaky_relu,prelu,elu,gelu}
                        Activation function for the transition MLP. (default:
                        tanh)

Regularization:
  --margin_threshold MARGIN_THRESHOLD
                        The desired difference between the best right and the
                        best wrong action. (default: 1.0)
  --error_probability ERROR_PROBABILITY
                        The probability to induce an error by choosing a wrong
                        action. (default: 0.1)
  --oov_probability OOV_PROBABILITY
                        A percentage to randomly replace tokens by the OOV
                        vector: freq / (freq + oov_prob). (default: 0.25)
  --update_frequency UPDATE_FREQUENCY
                        Update the weights after accumulating a certain number
                        of losses. (default: 50)
  --learning_rate LEARNING_RATE
                        The learning rate for the Adam trainer. (default:
                        0.001)
  --weight_decay WEIGHT_DECAY
                        Regularize the weights during an update. (default:
                        0.0)
  --gradient_clipping GRADIENT_CLIPPING
                        Make sure gradients do not get larger than this.
                        (default: 100.0)
  --token_dropout TOKEN_DROPOUT
                        Probability that any token will be replaced by an OOV
                        token. (default: 0.01)
  --lstm_dropout LSTM_DROPOUT
                        Dropout used between the stacked LSTMs. Note that
                        there is no support for recurrent dropout. (default:
                        0.33)
  --mlp_dropout MLP_DROPOUT
                        Dropout used between layers in the MLPs. (default:
                        0.25)
  --batch_size BATCH_SIZE
                        Number of sentences per batch. (default: 4)
  --loss_function {MaxMargin,CrossEntropy}
                        Name of the loss function to use. (default: MaxMargin)

Attention:
  --configuration_encoder {static,universal_attention,stack-buffer_query_attention,finished_tokens_attention,sentence_query_attention}
                        The method how to represent the current configuration
                        as input to the MLP. (default: static)
  --attention_reporter_path ATTENTION_REPORTER_PATH
                        Path to a folder where all the attention weights are
                        logged to. (default: None)
  --scale_query SCALE_QUERY
                        If set, scale the query vectors to this dimension.
                        (default: None)
  --scale_key SCALE_KEY
                        If set, scale the key vectors to this dimension.
                        (default: None)
  --scale_value SCALE_VALUE
                        If set, scale the value vectors to this dimension.
                        Must be equal to 'scale_key'. (default: None)
  --scoring_function {dot,scaled_dot,general,concat,learned,biaffine,dummy}
                        Name of the scoring function to use. (default: dot)
  --normalization_function {softmax,sigmoid,identity}
                        Name of the normalization function to use. (default:
                        softmax)
  --self_attention_heads SELF_ATTENTION_HEADS
                        Number of heads in the self-attention encoder if used.
                        The encoding dimensions must be dividable by this
                        number. (default: 10)

Misc.:
  --google_sheets_id GOOGLE_SHEETS_ID
                        The id of the Google Sheet to save the report in.
                        (default: None)
  --google_sheets_auth_path GOOGLE_SHEETS_AUTH_PATH
                        The auth.json file to for the Google API. (default:
                        None)
  --embeddings_vendor {glove,fasttext}
                        Name of the embeddings format. (default: glove)
  --freeze_embeddings FREEZE_EMBEDDINGS
                        Freeze the external embeddngs or not. (default: True)
  --show_progress_bars SHOW_PROGRESS_BARS
                        Show the progress bars for training and evaluation or
                        not. (default: True)
  --seed SEED           Number to initialize randomness with. (default: None)
  --experiment_name EXPERIMENT_NAME
                        Name of the experiment. Used for e.g. for logging.
                        (default: )
```