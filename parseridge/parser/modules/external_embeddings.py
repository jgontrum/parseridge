import numpy as np
import torch

from parseridge.utils.logger import LoggerMixin


class ExternalEmbeddings(LoggerMixin):

    def __init__(self, path, vendor="glove"):
        if vendor == "glove":
            self.token_to_embedding = self._load_glove(path)
        else:
            raise ValueError(f"Vendor '{vendor}' not supported.")

        self.dim = next(iter(self.token_to_embedding.values())).size

    @staticmethod
    def _load_glove(path):
        token_to_embedding = {}

        with open(path) as embedding_file:
            for line in embedding_file:
                token, *embeddings = line.split()
                token_to_embedding[token] = np.array([float(n) for n in embeddings])

        return token_to_embedding

    def get_weight_matrix(self, vocabulary, device="cpu"):
        embeddings = [
            np.random.rand(self.dim),           # OOV
            np.zeros(self.dim, dtype=float),    # PAD
            np.random.rand(self.dim),           # NUM
        ]

        for token, id_ in sorted(vocabulary._item_to_id.items(), key=lambda x: x[1]):
            if id_ <= 2:
                # Ignore <<OOV>> and <<PADDING>> vectors
                continue

            embeddings.append(self.token_to_embedding[token])

        np_embeddings = np.array(embeddings)

        token_embedding_weights = torch.from_numpy(np_embeddings).float().to(device)

        return torch.nn.Parameter(
            token_embedding_weights, requires_grad=True
        )

    @property
    def vocab(self):
        return set(self.token_to_embedding.keys())