# Training

The job of the `Trainer` is to train a given model using an `Optimizer` and the data in 
the `Corpus` objects inside a `Treebank`. We provide a base class `Trainer` which contains
the methods `fit_one_cycle()` to iterate over the corpus just once, `fit()` to learn
for multiple epochs, as well as `learn()`, which computes the loss and performs the 
back-propagation.
 
The `Trainer` is built around a callback system that allows seamless logging, evaluation,
and data manipulation without any need to modify the trainer itself.

!!! info "Callbacks"
    Have a look at TODO to see an overview over the different callback types and the
    already implemented callbacks.
    
## Dynamic Trainer
This trainer is the main implementation of the `Trainer` object and resembles the training
method in the original BiST- or UUParser. It parses the sentences in the `Corpus` using
a dynamic oracle to determine the order of configurations in each parsing process.

### Parsing a Sentence
A sentence is parsed incrementally until the terminal configuration is reached, where
the stack is empty and only the \*root\* token remains on the buffer. At the beginning,
the sentence is passed through the BiLSTM to contextualize the input tokens. In the parsing
process itself, the outputs of the LSTM are used to represent the tokens. At each parsing
step, the `Model` computes the logits for the transition that should be performed now, as
well as the relation label. 

### Batch Processing
Since the `DynamicTrainer` does not have training data in the traditional "Feature, Label"
format, the batch processing is more complicated. Keep in mind that a batch means a collection
of sentences to parse. During each parsing step, we compute a loss 