from collections import OrderedDict
from copy import copy

from parseridge.utils.logger import LoggerMixin


class Token(LoggerMixin):

    def __init__(self, id, form, head, deprel, misc="_", is_root=False, **kwargs):
        self.id = id
        self.form = form
        self.head = head
        self.relation = deprel
        self.is_root = is_root

        self.dependents = []
        self.parent = None
        self.projective_order = None

        self.lemma = kwargs.get("lemma")
        self.upostag = kwargs.get("upostag")
        self.xpostag = kwargs.get("xpostag")

        self.misc = misc

    @classmethod
    def create_root_token(cls):
        return cls(
            id=0,
            form="*root*",
            head=None,
            deprel="rroot",
            is_root=True
        )

    def get_unparsed_token(self):
        return Token(
            id=self.id,
            form=self.form,
            head=None,
            deprel=None,
            is_root=False,
            lemma=self.lemma,
            upostag=self.upostag,
            xpostag=self.xpostag,
            misc=self.misc
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
            "deps": "_",
            "misc": self.misc
        }
        for k, v in copy(serialized).items():
            if v is None:
                v = "_"
            serialized[k] = v

        return OrderedDict(serialized)

    def __repr__(self):
        return "\t".join([str(v) for v in self.serialize().values()])
