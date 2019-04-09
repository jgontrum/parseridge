import math
import random
from copy import deepcopy
from itertools import chain

import numpy as np
import torch
from torch.utils.data import Dataset

from parseridge.utils.logger import LoggerMixin


class Corpus(Dataset, LoggerMixin):

    def __init__(self, sentences, vocabulary, device="cpu"):
        """
        A Corpus stores all the sentences and their numeric representations.
        Use `get_iterator()` to generate an iterator over the data.

        Note: This class implements PyTorch's Dataset class, however
        it is advised to use the custom iterator instead, because PyTorch's
        Dataloader only supports tensors, we need, however, to return
        Sentence objects as well.

        Parameters
        ----------
        sentences : list of Sentence objects
            The sentences for this corpus.
        vocabulary : Signature object
            Converts sentence tokens into integers. If this is not a training
            corpus, it is advised to set the Signature object into a read-only
            mode to avoid dimension mismatches in the embeddings.
        device : str
            The PyTorch device to copy the sentence representations to.
        """
        self.vocabulary = vocabulary
        self.sentences = sentences
        self.device = device

        self.sentence_tensors = []
        self.sentence_token_freq_tensors = []
        self.sentence_lengths = [len(sentence) for sentence in sentences]
        self.sorted_indices = np.argsort(self.sentence_lengths)
        self.num_oov_tokens = 0

        assert self.vocabulary.get_id("<<<PADDING>>>") == 1

        # Count words, we need them later to blank out seldom word to train the OOV
        # word embedding.
        [self.vocabulary.add(token.form) for sentence in sentences for token in sentence]

        # Add ID to sentences so that we can reconstruct the order later
        self.logger.info(f"Loading {len(self.sentences)} sentences...")
        for i, sentence in enumerate(self.sentences):
            sentence.id = i
            tokens, frequencies = self._prepare_sentence(sentence)
            self.sentence_tensors.append([tokens])
            self.sentence_token_freq_tensors.append([frequencies])

        # Copy sentence representations to device memory
        self.logger.info(f"The corpus contains {self.num_oov_tokens} OOV tokens.")
        self.logger.info("Copying sentence representation to device memory...")
        self.sentence_tensors = torch.tensor(
            self.sentence_tensors, dtype=torch.long, device=self.device)
        self.sentence_token_freq_tensors = torch.tensor(
            self.sentence_token_freq_tensors, dtype=torch.float, device=self.device)
        self.logger.info("Done!")

    def _pad_list(self, list_, max_sentence_length):
        """
        Pad the rest of the list with zeros.
        Parameters
        ----------
        list_ : List of int
            List to pad.
        max_sentence_length : int
            Desired length of the list.
        Returns
        -------
        List of int
        """
        pad_size = max_sentence_length - len(list_)
        padding = np.ones(pad_size)
        return np.concatenate((list_, padding))

    def _prepare_sentence(self, sentence):
        """
        Replaces the tokens in the sentence by integers and pads the output.
        This is the place to add more features in the future like characters
        or part-of-speech tags.

        Parameters
        ----------
        sentence : Sentence object

        Returns
        -------
        List of int
        """
        tokens = [self.vocabulary.get_id(token.form) for token in sentence]
        self.num_oov_tokens += tokens.count(self.vocabulary.get_id("<<<OOV>>>"))

        sentence_padded = self._pad_list(tokens, max(self.sentence_lengths))

        token_frequencies = [self.vocabulary.get_count(token.form) for token in sentence]
        frequencies_padded = self._pad_list(token_frequencies, max(self.sentence_lengths))

        return sentence_padded, frequencies_padded

    def __len__(self):
        return len(self.sentence_lengths)

    def __getitem__(self, index):
        return self.sentence_tensors[index]


class CorpusIterator(LoggerMixin):

    def __init__(self, corpus, batch_size=8, shuffle=False, drop_last=False, train=False,
                 oov_probability=0.25, group_by_length=True, token_dropout=0.1):
        """
        Helper class to iterate over the batches produced by the Corpus class.
        Most importantly, it has the ability to shuffle the order of the batches.
        This helper is needed because the corpus returns not only a Tensor with
        the numeric representation of a sentence, but also the sentence object
        itself, which is not supported by PyTorch's DataLoader class.

        Parameters
        ----------
        corpus
        batch_size
        shuffle
        drop_last
        """
        self.corpus = corpus
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.oov_probability = oov_probability
        self._iter = 0

        assert self.corpus.vocabulary.get_id("<<<OOV>>>") == 0
        assert self.corpus.vocabulary.get_id("<<<PADDING>>>") == 1

        self.sentence_tensors = self.corpus.sentence_tensors
        self._num_batches = len(self.corpus) / self.batch_size

        # Replace the ids of some infrequent words randomly with the OOV id to train
        # the OOV embedding vector.
        if train and oov_probability > 0:
            self.sentence_tensors = self.replace_infrequent_words_with_oov(
                self.sentence_tensors,
                self.corpus.sentence_token_freq_tensors,
                self.oov_probability
            )

        # As a regularization technique, we randomly replace tokens with the OOV id.
        # In contrast to the OOV handling, this can affect all words.
        # Note: The percentage of dropped out tokens is smaller than the dropout
        # probability, as it is applied to the whole data, including the padding.
        if train and token_dropout > 0:
            self.sentence_tensors = self.apply_token_dropout(
                self.sentence_tensors, p=token_dropout)

        # If len(self.corpus) % self.batch_size != 0, one batch will be slightly
        # larger / smaller than the other ones. Use drop_last to ignore this one batch.
        if self.drop_last:
            self._num_batches = math.floor(self._num_batches)
        else:
            self._num_batches = math.ceil(self._num_batches)

        # When using a batch_size > 1, performance and possibly accuracy can be improved
        # by grouping sentences with similar length together to make better use of the
        # batch processing. To do so, the content of all batches will be static,
        # but their order will be randomized if shuffle is activated.
        if group_by_length:
            self._order = self.group_batches_by_length(
                self.corpus.sentences, self.batch_size, self.shuffle
            )
        else:
            # The naive way: Take the ids of all sentences and randomize them if wanted.
            self._order = list(range(len(self.corpus)))

            if self.shuffle:
                random.shuffle(self._order)

    @staticmethod
    def replace_infrequent_words_with_oov(sentence_tensors, frequency_tensors,
                                          oov_probability):
        # Compute the relative frequency
        oov_probability_tensor = torch.zeros_like(frequency_tensors).fill_(oov_probability)
        frequency_tensors = frequency_tensors / (frequency_tensors + oov_probability_tensor)

        rand = torch.rand_like(sentence_tensors, dtype=torch.float)
        mask = torch.lt(rand, frequency_tensors).type(torch.long)
        return torch.mul(sentence_tensors, mask)

    @staticmethod
    def apply_token_dropout(sentence_tensors, p):
        dropout = torch.rand_like(sentence_tensors, dtype=torch.float).fill_(p)

        rand = torch.rand_like(sentence_tensors, dtype=torch.float)
        mask = torch.lt(dropout, rand).type(torch.long)
        return torch.mul(sentence_tensors, mask)

    @staticmethod
    def group_batches_by_length(sentences, batch_size, shuffle):
        sentences_sorted = [
            sentence.id for sentence in
            sorted(sentences, key=lambda s: len(s))
        ]

        # Make the list dividable by batch_size
        rest_size = len(sentences_sorted) % batch_size
        rest = sentences_sorted[-rest_size:]
        order = sentences_sorted[:-rest_size]

        chunks = np.array_split(order, len(order) / batch_size)

        if shuffle:
            random.shuffle(chunks)

        return list(chain(*chunks)) + rest

    def __len__(self):
        return self._num_batches

    def __iter__(self):
        return self

    def __next__(self):
        if self._iter >= len(self):
            raise StopIteration
        else:
            start = self._iter * self.batch_size
            indices = self._order[start:start + self.batch_size]

            # Ignore an incomplete batch at the end if wished
            if len(indices) < self.batch_size and self.drop_last:
                raise StopIteration

            batch_sentences = [self.corpus.sentences[i] for i in indices]

            # Sort the indices in descending order - this is required for
            # batch processing in PyTorch.
            batch_sentences = sorted(
                batch_sentences, key=lambda s: len(s), reverse=True
            )
            indices_sorted = [sentence.id for sentence in batch_sentences]

            batch_tensors = self.sentence_tensors[indices_sorted]

            # Cut of unnecessary padding
            longest_sentence = max([len(s) for s in batch_sentences])
            batch_tensors = batch_tensors[:, :, :longest_sentence]

            self._iter += 1
            return batch_tensors, deepcopy(batch_sentences)
