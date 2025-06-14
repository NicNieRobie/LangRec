from typing import Optional

import torch
from sentence_transformers import SentenceTransformer

from model.base_model import BaseModel


class SentenceBertModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model = SentenceTransformer(self.key)
        self.model.get_sentence_embedding_dimension()

    def embed(self, content, func=None, truncate=None) -> Optional[torch.Tensor]:
        return self.model.encode(content, convert_to_tensor=True).float().cpu().detach().numpy()


class SentenceT5Model(SentenceBertModel):
    pass
