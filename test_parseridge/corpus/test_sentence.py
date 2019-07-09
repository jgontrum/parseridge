from parseridge.corpus.sentence import Sentence
from test_parseridge.utils import get_fixtures_path, log_stderr


@log_stderr
def test_read_and_serialize():
    with open(get_fixtures_path("sentence_01.conllu")) as sentence_io:
        train_sentence = list(Sentence.from_conllu("".join(sentence_io)))[0]

    assert (
        repr(train_sentence)
        == """# newdoc id = GUM_academic_art
# sent_id = GUM_academic_art-1
# text = Aesthetic Appreciation and Spanish Art:
# s_type = frag
1	Aesthetic	aesthetic	ADJ	JJ	_	2	amod	_	_
2	Appreciation	appreciation	NOUN	NN	_	0	root	_	_
3	and	and	CCONJ	CC	_	5	cc	_	_
4	Spanish	Spanish	ADJ	JJ	_	5	amod	_	_
5	Art	art	NOUN	NN	_	2	conj	_	SpaceAfter=No
6	:	:	PUNCT	:	_	2	punct	_	_

"""
    )
