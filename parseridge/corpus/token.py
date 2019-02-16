from collections import OrderedDict
from copy import copy


class Token:

    def __init__(self, id, form, head, deps, deprel, is_root=False, **kwargs):
        self.id = id
        self.form = form
        self.head = head
        self.deps = deps
        self.relation = deprel
        self.is_root = is_root

        self.dependents = []
        self.parent = None
        self.projective_order = None

        self.lemma = kwargs.get("lemma")
        self.upostag = kwargs.get("upostag")
        self.xpostag = kwargs.get("xpostag")

    @classmethod
    def create_root_token(cls):
        return cls(
            id=0,
            form="<<<ROOT>>>",
            head=None,
            deps=None,
            deprel="rroot",
            is_root=True
        )

    def get_unparsed_token(self):
        return Token(
            id=self.id,
            form=self.form,
            head=None,
            deps=None,
            deprel=None,
            is_root=False,
            lemma=self.lemma,
            upostag=self.upostag,
            xpostag=self.xpostag
        )

    def serialize(self):
        serialized = {
            "id": self.id,
            "form": self.form,
            "lemma": self.lemma,
            "upostag": self.upostag,
            "xpostag": self.xpostag,
            "feats": "_",
            "head": self.head,
            "deprel": self.relation,
            "deps": self.deps,
            "misc": "_"
        }
        for k, v in copy(serialized).items():
            if v is None:
                v = "_"
            serialized[k] = v

        return OrderedDict(serialized)

    def __repr__(self):
        return "\t".join([str(v) for v in self.serialize().values()])
