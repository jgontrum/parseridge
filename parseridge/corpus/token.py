from collections import OrderedDict
from copy import copy
from typing import Optional, List

from parseridge.utils.logger import LoggerMixin


class Token(LoggerMixin):
    def __init__(
        self,
        id: int,
        form: str,
        head: Optional[int],
        deprel: Optional[str],
        misc: str = "_",
        is_root: bool = False,
        **kwargs,
    ) -> None:
        self.id = id
        self.form = form
        self.head = head
        self.relation = deprel
        self.is_root = is_root

        self.dependents: List[int] = []
        self.parent = None
        self.projective_order = None

        self.lemma = kwargs.get("lemma")
        self.upostag = kwargs.get("upostag")
        self.xpostag = kwargs.get("xpostag")

        self.misc = misc

    @classmethod
    def create_root_token(cls) -> "Token":
        return cls(id=0, form="*root*", head=None, deprel="rroot", is_root=True)

    def get_unparsed_token(self) -> "Token":
        return Token(
            id=self.id,
            form=self.form,
            head=None,
            deprel=None,
            is_root=False,
            lemma=self.lemma,
            upostag=self.upostag,
            xpostag=self.xpostag,
            misc=self.misc,
        )

    def serialize(self) -> OrderedDict:
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
            "misc": self.misc,
        }

        for k, v in copy(serialized).items():
            if v is None:
                v = "_"
            serialized[k] = v

        return OrderedDict(serialized)

    def __repr__(self) -> str:
        return "\t".join([str(v) for v in self.serialize().values()])
