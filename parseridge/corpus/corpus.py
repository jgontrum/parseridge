import math
import random
from copy import deepcopy

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
            self.sentence_tensors, dtype=torch.long).to(self.device)
        self.sentence_token_freq_tensors = torch.tensor(
            self.sentence_token_freq_tensors, dtype=torch.float).to(self.device)
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
        tokens = [self.vocabulary.add(token.form) for token in sentence]
        self.num_oov_tokens += tokens.count(self.vocabulary.get_id("<<<OOV>>>"))

        sentence_padded = self._pad_list(tokens, max(self.sentence_lengths))

        token_frequencies = [self.vocabulary.get_count(token.form) for token in sentence]
        token_frequencies = [freq / (0.25 + freq) for freq in token_frequencies]
        frequencies_padded = self._pad_list(token_frequencies, max(self.sentence_lengths))

        return sentence_padded, frequencies_padded

    def get_iterator(self, batch_size=32, shuffle=True, drop_last=False, train=False):
        """
        Returns an `CorpusIterator` object that is used to iterate over the
        corpus using the given parameters.

        Parameters
        ----------
        batch_size : int
            Size of the batches that the iterator outputs.
        shuffle : bool
            Whether the order of the sentences is randomized.
        drop_last : bool
            If the last batch is smaller than `batch_size`, ignore it.

        Returns
        -------
        CorpusIterator object
        """
        return CorpusIterator(self, batch_size, shuffle, drop_last, train)

    def __len__(self):
        return len(self.sentence_lengths)

    def __getitem__(self, index):
        return self.sentence_tensors[index]


class CorpusIterator(LoggerMixin):

    def __init__(self, corpus, batch_size=48, shuffle=False, drop_last=False, train=False):
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
        self._iter = 0

        assert self.corpus.vocabulary.get_id("<<<OOV>>>") == 0
        assert self.corpus.vocabulary.get_id("<<<PADDING>>>") == 1

        self.sentence_tensors = self.corpus.sentence_tensors

        if False:
            frequency_tensors = self.corpus.sentence_token_freq_tensors
            rand = frequency_tensors.data.new(frequency_tensors.size()).uniform_()
            mask = torch.lt(rand, frequency_tensors).type(torch.long)

            from functools import reduce
            oov = int((reduce(lambda a, b: a * b, mask.size()) - torch.sum(mask)).item())
            self.logger.info(f"Corpus iteration contains {oov} replaced OOV tokens.")
            self.sentence_tensors = torch.mul(self.sentence_tensors, mask)

        self._num_batches = len(self.corpus) / self.batch_size
        if self.drop_last:
            self._num_batches = math.floor(self._num_batches)
        else:
            self._num_batches = math.ceil(self._num_batches)

        self._order = list(range(len(self.corpus)))
        if self.shuffle:
            random.shuffle(self._order)

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
