# Parseridge
A Transition-based Dependency Parser in PyTorch.

## Usage
```bash
usage: main.py [-h] --train_corpus TRAIN_CORPUS --test_corpus TEST_CORPUS
               [--num_stack NUM_STACK] [--num_buffer NUM_BUFFER]
               [--embedding_size EMBEDDING_SIZE]
               [--lstm_hidden_size LSTM_HIDDEN_SIZE]
               [--lstm_layers LSTM_LAYERS]
               [--relation_mlp_layers RELATION_MLP_LAYERS [RELATION_MLP_LAYERS ...]]
               [--transition_mlp_layers TRANSITION_MLP_LAYERS [TRANSITION_MLP_LAYERS ...]]
               [--margin_threshold MARGIN_THRESHOLD]
               [--error_probability ERROR_PROBABILITY]
               [--oov_probability OOV_PROBABILITY] [--update_size UPDATE_SIZE]
               [--loss_factor LOSS_FACTOR] [--loss_strategy LOSS_STRATEGY]
               [--learning_rate LEARNING_RATE] [--weight_decay WEIGHT_DECAY]
               [--gradient_clipping GRADIENT_CLIPPING]
               [--token_dropout TOKEN_DROPOUT] [--lstm_dropout LSTM_DROPOUT]
               [--mlp_dropout MLP_DROPOUT] [--batch_size BATCH_SIZE]
               [--comment COMMENT] [--seed SEED] [--epochs EPOCHS]
               [--device DEVICE] [--train]

optional arguments:
  -h, --help            show this help message and exit
  --comment COMMENT     A comment about this experiment. (default: )
  --seed SEED           Number to initialize randomness with. (default: None)
  --epochs EPOCHS       Number of epochs to run. (default: 30)
  --device DEVICE       Device to run on. cpu or cuda. (default: cpu)
  --train               Use in training mode. (default: True)

Files:
  --train_corpus TRAIN_CORPUS
                        Path to train file. (default: None)
  --test_corpus TEST_CORPUS
                        Path to test file. (default: None)

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
  --lstm_hidden_size LSTM_HIDDEN_SIZE
                        Size of the hidden layer of the LSTM. The output size
                        will be 2x the size. (default: 125)
  --lstm_layers LSTM_LAYERS
                        Number of stacked LSTMs (default: 2)
  --relation_mlp_layers RELATION_MLP_LAYERS [RELATION_MLP_LAYERS ...]
                        List of sizes of the layers in the MLP for the
                        relation labels. (default: [100])
  --transition_mlp_layers TRANSITION_MLP_LAYERS [TRANSITION_MLP_LAYERS ...]
                        List of sizes of the layers in the MLP for the
                        transitions. (default: [100])

Regularization:
  --margin_threshold MARGIN_THRESHOLD
                        The desired difference between the best right and the
                        best wrong action. (default: 2.5)
  --error_probability ERROR_PROBABILITY
                        The probability to induce an error by choosing a wrong
                        action. (default: 0.1)
  --oov_probability OOV_PROBABILITY
                        A percentage to randomly replace tokens by the OOV
                        vector: freq / (freq + oov_prob). (default: 0.25)
  --update_size UPDATE_SIZE
                        Update the weights after accumulating a certain number
                        of losses. (default: 50)
  --loss_factor LOSS_FACTOR
                        Multiply the accumulated loss with this number to
                        regularize it. (default: 0.75)
  --loss_strategy LOSS_STRATEGY
                        Strategy to reduce a list of loss values to one.
                        Supported are avg and sum. (default: avg)
  --learning_rate LEARNING_RATE
                        The learning rate for the Adam trainer. (default:
                        0.001)
  --weight_decay WEIGHT_DECAY
                        Regularize the weights during an update. (default:
                        0.0)
  --gradient_clipping GRADIENT_CLIPPING
                        Make sure gradients do not get larger than this.
                        (default: 10.0)
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
```