import numpy as np
import torch

from parseridge.corpus.corpus import CorpusIterator
from parseridge.corpus.treebank import Treebank
from parseridge.parser.attention_model import AttentionModel
from parseridge.utils.helpers import set_seed
from test_parseridge.utils import log_stderr, get_fixtures_path


def generate_dummy_sentence(length, size, offset=0):
    return torch.stack(tuple([
        torch.rand((size,)).fill_(float(f"{i + 1}.{offset}"))
        for i in range(length)
    ]))


def generate_sentence_batch(batch_size, length, size):
    sentence_batch = []
    for i in range(batch_size):
        sentence_batch.append(
            generate_dummy_sentence(length, size, offset=i)
        )
    return torch.stack(tuple(sentence_batch))


class TestParseProjective:

    @classmethod
    def setup_class(cls):
        cls.embedding_size = 5
        cls.batch_size = 3
        with open(get_fixtures_path("sentence_batch_l3.conllu")) as train_io:
            treebank = Treebank(
                train_io=train_io,
                dev_io=None,
                device="cpu"
            )

        cls.corpus = treebank.train_corpus
        cls.vocabulary = treebank.vocabulary
        cls.relations = treebank.relations

        cls.model = AttentionModel(
            relations=cls.relations,
            vocabulary=cls.vocabulary,
            embedding_size=cls.embedding_size,
            device="cpu"
        )

    @log_stderr
    def test_tensor_for_indices(self):
        batch_size = 3
        embedding_size = 5
        num_words = 10
        size = 3

        # Generate the LSTM output in the size batch x num_words x embedding_size
        sentence_batch = generate_sentence_batch(batch_size, num_words, embedding_size)

        padding = torch.rand((embedding_size,)).fill_(0.0)

        stacks = [
            [0, 1],
            [1, 2, 3],
            []
        ]

        output = AttentionModel.lookup_tensors_for_indices(
            indices_batch=stacks,
            sequence_batch=sentence_batch,
            padding=padding,
            size=size
        )

        assert output.is_contiguous()

        stack_1_output = output[0].numpy()
        expected_output_1 = [
            1.0, 1.0, 1.0, 1.0, 1.0,  # token 1
            2.0, 2.0, 2.0, 2.0, 2.0,  # token 2
            0.0, 0.0, 0.0, 0.0, 0.0,  # padding
        ]
        assert np.array_equal(stack_1_output,
                              np.array(expected_output_1, dtype=np.float32))

        stack_2_output = output[1].numpy()
        expected_output_2 = [
            2.1, 2.1, 2.1, 2.1, 2.1,  # token 2
            3.1, 3.1, 3.1, 3.1, 3.1,  # token 3
            4.1, 4.1, 4.1, 4.1, 4.1,  # token 4
        ]
        assert np.array_equal(stack_2_output,
                              np.array(expected_output_2, dtype=np.float32))

        stack_3_output = output[2].numpy()
        expected_output_3 = [
            0.0, 0.0, 0.0, 0.0, 0.0,  # padding
            0.0, 0.0, 0.0, 0.0, 0.0,  # padding
            0.0, 0.0, 0.0, 0.0, 0.0,  # padding
        ]
        assert np.array_equal(stack_3_output,
                              np.array(expected_output_3, dtype=np.float32))

    @log_stderr
    def test_stack_buffer_concatenation(self):
        batch_size = 3
        embedding_size = 5
        num_words = 10
        stack_size = 3
        buffer_size = 1

        # Generate the LSTM output in the size batch x num_words x embedding_size
        sentence_batch = generate_sentence_batch(batch_size, num_words, embedding_size)

        padding = torch.rand((embedding_size,)).fill_(0.0)

        stacks = [
            [0, 1],
            [1, 2, 3],
            []
        ]

        stack_batch = AttentionModel.lookup_tensors_for_indices(
            indices_batch=stacks,
            sequence_batch=sentence_batch,
            padding=padding,
            size=stack_size
        )

        buffers = [
            [0],
            [1],
            []
        ]

        buffer_batch = AttentionModel.lookup_tensors_for_indices(
            indices_batch=buffers,
            sequence_batch=sentence_batch,
            padding=padding,
            size=buffer_size
        )

        output = AttentionModel._concatenate_stack_and_buffer(stack_batch, buffer_batch)

        assert [int(n) for n in output.size()] == \
               [batch_size, (stack_size + buffer_size) * embedding_size]

        batch_1_output = output[0].numpy()
        expected_output_1 = [
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000,  # stack w2
            2.0000, 2.0000, 2.0000, 2.0000, 2.0000,  # stack w3
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,  # padding
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000,  # buffer w
        ]
        assert np.array_equal(batch_1_output,
                              np.array(expected_output_1, dtype=np.float32))

        batch_2_output = output[1].numpy()
        expected_output_2 = [
            2.1000, 2.1000, 2.1000, 2.1000, 2.1000,  # stack w2
            3.1000, 3.1000, 3.1000, 3.1000, 3.1000,  # stack w3
            4.1000, 4.1000, 4.1000, 4.1000, 4.1000,  # stack w4
            2.1000, 2.1000, 2.1000, 2.1000, 2.1000,  # buffer w2
        ]
        assert np.array_equal(batch_2_output,
                              np.array(expected_output_2, dtype=np.float32))

        batch_3_output = output[2].numpy()
        expected_output_3 = [
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,  # padding
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,  # padding
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,  # padding
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,  # padding
        ]
        assert np.array_equal(batch_3_output,
                              np.array(expected_output_3, dtype=np.float32))

    @log_stderr
    def test_lstm_output(self):
        set_seed(0)

        it = CorpusIterator(
            self.corpus, batch_size=3, shuffle=False, drop_last=False, train=False,
            group_by_length=False
        )

        sentence_features, sentences = next(it)
        assert np.array_equal(sentence_features.numpy(), np.array(
            # 2 = root token, 1 = padding token
            [
                [[2, 12, 13, 14, 15, 16, 17, 18, 19, 20]],
                [[2, 3, 4, 5, 6, 7, 8, 1, 1, 1]],
                [[2, 9, 10, 11, 1, 1, 1, 1, 1, 1]]
            ],
            dtype=np.int64
        ))

        output = self.model.compute_lstm_output(sentences, sentence_features)
        assert [int(n) for n in output.size()] == \
               [
                   len(sentences),
                   max([len(s) for s in sentences]),
                   2 * self.model.lstm_hidden_size
               ]
