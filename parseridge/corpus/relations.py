from parseridge.corpus.signature import Signature
from parseridge.utils.helpers import Relation, T


class Relations:

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

        self.num_relations = len(self.labels)

        self.slices = {
            T.SHIFT: slice(0, 1),
            T.SWAP: slice(1, 2),
            T.LEFT_ARC: slice(2, self.num_relations, 2),
            T.RIGHT_ARC: slice(3, self.num_relations, 2),
        }
