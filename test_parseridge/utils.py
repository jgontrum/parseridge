import logging

from torch import nn

from parseridge.utils.helpers import Action


def log_stderr(func):
    """Decorator for redirecting loggers to stderr."""

    def inner(*args, **kwargs):
        logging.getLogger().handlers = [logging.StreamHandler()]
        return func(*args, **kwargs)

    return inner


def get_fixtures_path(file_name):
    return f"test_parseridge/fixtures/{file_name}"


def tokens_are_equal(token1, token2):
    if token1 is None or token2 is None:
        assert token1 == token2
    else:
        assert token1.id == token2.id, f"'{token1.id}' != '{token2.id}'"
        assert token1.form == token2.form, f"'{token1.form}' != '{token2.form}'"
        assert token1.head == token2.head, f"'{token1.head}' != '{token2.head}'"
        assert token1.relation == token2.relation, f"'{token1.relation}' != '{token2.relation}'"
        assert token1.is_root == token2.is_root, f"'{token1.is_root}' != '{token2.is_root}'"
        assert token1.dependents == token2.dependents, f"'{token1.dependents}' != '{token2.dependents}'"
        assert token1.lemma == token2.lemma, f"'{token1.lemma}' != '{token2.lemma}'"
        assert token1.upostag == token2.upostag, f"'{token1.upostag}' != '{token2.upostag}'"
        assert token1.xpostag == token2.xpostag, f"'{token1.xpostag}' != '{token2.xpostag}'"
        tokens_are_equal(token1.parent, token2.parent)
    return True


def sentences_are_equal(sentence1, sentence2):
    assert len(sentence1) == len(sentence2)
    for token1, token2 in zip(sentence1, sentence2):
        assert tokens_are_equal(token1, token2)
    return True


def generate_actions(transition, relations, best_relations):
    return [
        Action(relation, transition, float(relation in best_relations))
        for relation in relations
    ]


def set_weights_to_one(model):
    for name, param in model.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            nn.init.constant_(param, 1.0)
