# Data Preparation

This section describes the process of reading in treebanks and other input data, 
as well as highlighting the way this data represented internally.

## Treebank

The [`Treebank`](#treebank) class is mostly a wrapper around [`Corpus`](#corpus) objects for the train, develop, and test sets.
Additionally, it also manages the [`Signature`](#base-signature) objects used to map tokens, relations, and 
other categorial data to their internal ids.

For training, a [`Treebank`](#treebank) object can be initialized as follows:

```python
from parseridge.corpus.treebank import Treebank

treebank = Treebank(
    train_io=open("data/UD_English-GUM/en_gum-ud-train.conllu"),
    dev_io=open("data/UD_English-GUM/en_gum-ud-dev.conllu"),
    test_io=open("data/UD_English-GUM/en_gum-ud-test.conllu")  # optional,
)
```

## Signatures

### Base Signature
The default [`Signature`](#base-signature) class manages a mapping between objects and their integer ids. 
They are build by adding new items to the mapping using the `add()` method. If the item
already exists, its current id is returned, if not, it is assigned a new id. The object
can later be accessed via its `get_id(item)` and `get_item(id)` methods.

### Vocabulary
The [`Vocabulary`](#vocabulary) is a special kind of [`Signature`](#base-signature). It is used to map the strings of the
tokens in the corpus to their ids. It is initialized with the entries for the OOV-token,
the padding token, an entry for numeric tokens ('NUM'), as well as the root token (\*root\*).

By default, all tokens are automatically lowercased and numbers (e.g. '12.123') are mapped
to the 'NUM' token.

Additionally, the [`Vocabulary`](#vocabulary) can also be initialized with a set of tokens with pre-trained
embeddings. In this case, the unchanged string of the given token is matched against the 
tokens with embeddings and the OOV id is returned in case no embedding exists for it:

```python
from parseridge.corpus.vocabulary import Vocabulary

v = Vocabulary(embeddings_vocab=set("word1, word2"))
v.add("word1")  # id = 3
v.add("word2")  # id = 4
v.add("word3")  # id = 0 - the OOV token
```

### Relations
The [`Relations`](#relations) class does not inherit from the [`Signature`](#base-signature) class, but uses it as an
internal object.
It manages both the mapping of relation labels (e.g. 'nsubj') in the the [`Signature`](#base-signature) object,
as well as the mapping of `Relation` object - a transition + a label - in the `label_signature`
object.

Both signatures are used in different contexts: For one, we treat the combination of an ARC
transition with a relation label as a unit we predict on our labeled MLP. On the other hand, 
we also must be abel to interpret the strings of the relation labels independently, 
for example when checking weather a label was correctly predicted.

!!! info "Separate Relation Signatures"
    This behaviour is explained in more detail in the section about the parsing algorithm. 

The object must be initialized with a list of [`Sentence`](#sentence) objects from which it extracts
all labels.

## Sentence
[`Sentence`](#sentence) objects are usually created by parsing a treebank in CoNLL-U format using the
`Sentence.from_conllu(str)` method. They can be treated like an iterable over the tokens
in the sentence, which the [`Sentence`](#sentence) object enriches during initialization. For example,
it assigns each [`Token`](#token) its projective order, as well as the parent [`Token`](#token) or a list of
dependents. The [`Sentence`](#sentence) also contains meta information about the sentence like the 
original text or its id. The first token in each sentence is the root token.

```python
from parseridge.corpus.sentence import Sentence

sentences = list(Sentence.from_conllu("".join(open("treebank.conllu"))))
```

## Token
A [`Token`](#token) is mostly a storage class that holds information like its id (from the [`Vocabulary`](#vocabulary)),
head and dependents, as well as part-of-speech information

## Corpus
The [`Corpus`](#corpus) class is used to convert a list of [`Sentence`](#sentence) objects into a PyTorch tensor.
During the initialization, it adds all words in the sentences to the signatures, pads the
numeric representations, and eventually copies the tensor to the device memory (CPU or GPU).
Internally, the corpus is represented as one large padded tensor with the following layout:

|Dimension|Length              |Content|
|:-------:|--------------------|-------|
|0        |Number of sentences |A sentence tensor|
|1        |Number of features  |A feature representation, e.g. token ids or part-of-speech tag ids|
|2        |Max. sentence length|The integer representation of a feature|


The different features are stored in the following dimensions:

|Dimension|Feature|Implemented|
|:-------:|-------|:---------:|
|0        |Token id | Yes |
|1        |Part-of-speech tag id| No|
|2        |Character-based id   | No|
|3        |Contextualized token id   | No|


The corpus also computes the overall frequency of the tokens in the corpus, which are used
later on to randomly drop infrequent works to better train the embedding for the 
out-of-vocabulary token.

### Corpus Iterator
As the name suggests, the purpose of the [`CorpusIterator`](#corpus-iterator) is to iterate over the sentences
in a [`Corpus`](#corpus) object during training or evaluation. It selects sentences from the corpus,
merges them into a new tensor object and cuts the corpus-level padding if needed.

It can be initialized with the following parameters:

|Parameter|Description|Default|
|---------|-----------|:-----:|
|[`Corpus`](#corpus) |The [`Corpus`](#corpus) object to iterate|*required*|
|`batch_size`|Number of sentences per batch|8|
|`shuffle`|Should the order of sentences be random?|false|
|`drop_last`|If the last batch is smaller than the batch size, should it be dropped?|false|
|`train`|If set true, regularization is applied. Use only for training, not prediction.|false|
|`oov_probability`|Regularization. Drops infrequent tokens randomly|0.25|
|`group_by_length`|Should a batch consist of sentence of similar length?|true|
|`token_dropout`|Regularization. Drops any token randomly|0.1|


!!! info "Infrequent Tokens Regularization"
    The term `oov_probability` is a bit misleading, as the value does not directly
    describe a probability. Instead, we decide weather to randomly replace a token
    with the OOV token or not using the following calculation:
    $$
    r(t) = \frac{t_f}{t_f + p},
    $$
    
    where $t_f$ is the frequency of the token $t$ and $p$ the OOV probability.