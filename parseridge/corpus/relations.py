from parseridge.corpus.signature import Signature
from parseridge.utils.helpers import Relation, T
from parseridge.utils.logger import LoggerMixin


class Relations(LoggerMixin):

    def __init__(self, sentences):
        relations = set()
        for sentence in sentences:
            for token in sentence:
                if token.relation:
                    relations.add(token.relation)

        relations = list(sorted(relations))

        self.labels = [
            Relation(T.SHIFT, None),
            Relation(T.SWAP, None),
        ]

        self.signature = Signature(warn_on_oov=True)

        for relation in relations:
            self.signature.add(relation)
            self.labels.append(Relation(T.LEFT_ARC, relation))
            self.labels.append(Relation(T.RIGHT_ARC, relation))

        self.slices = {
            T.SHIFT: slice(0, 1),
            T.SWAP: slice(1, 2),
            T.LEFT_ARC: slice(2, len(self.labels), 2),
            T.RIGHT_ARC: slice(3, len(self.labels), 2),
        }

    @property
    def relations(self):
        return self.signature.get_items()

    def __len__(self):
        return len(self.labels)
