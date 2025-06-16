from typing import Optional

import torch
from torch import nn

from loader.code_map import CodeMap as Map
from model.base_dense_code_model import DenseCodeEmbeddingLayer, BaseDenseCodeModel
from model.base_model import BaseModel


class DiscreteCodeEmbeddingLayer(DenseCodeEmbeddingLayer):
    def __init__(self, num_codes: int, dtype, **kwargs):
        super().__init__(**kwargs)

        self.code_embeddings = nn.Embedding(num_codes, self.embedding_dim, dtype=dtype)
        self.code_embeddings.weight.requires_grad = True

        self.code_classifier = nn.Linear(self.embedding_dim, num_codes, bias=False, dtype=dtype)

    def classify(self, embeds):
        return self.code_classifier(embeds)


class BaseDiscreteCodeModel(BaseDenseCodeModel):
    def __init__(self, num_codes, **kwargs):
        super().__init__(**kwargs)
        self.embedding_layer: Optional[DiscreteCodeEmbeddingLayer] = None
        self.embedding_dimension = self.get_token_embeddings().weight.shape[1]
        self.num_codes = num_codes

    def load(self):
        BaseModel.load(self)

        self.embedding_layer = DiscreteCodeEmbeddingLayer(
            llm_embeddings=self.get_token_embeddings(),
            device=self.device,
            num_codes=self.num_codes,
            dtype=self.get_dtype(),
        )
        self.embedding_layer.to(self.device)

        return self

    def set_code_embeddings(self, code_embeddings):
        raise AttributeError('set_code_embeddings is not supported in DiscreteCodeModel')

    def _get_scores(self, batch):
        output = self.embedding_layer(batch)

        input_embeddings = output['input_embeddings']
        attention_mask = output['attention_mask']

        code_input = output['code_input']  # [B, L]
        code_mask = output['code_mask']  # [B, L]

        output = self.model(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        logits = output.logits  # [B, L, V]
        indices = (batch[Map.LEN_COL] - 1).view(-1, 1, 1).expand(-1, 1, logits.size(-1)).to(self.device)
        logits = torch.gather(logits, 1, indices).squeeze(1)  # [B, V]
        logits = logits[:, self.label_tokens]  # [B, 2]
        scores = self.softmax_sft(logits)  # [B, 2]

        states = output.hidden_states[-1]  # [B, L, D]
        logits = self.embedding_layer.classify(states)  # [B, L, C]

        # left shift code input and mask to construct the target
        code_input = torch.roll(code_input, -1, 1)  # [B, L]
        code_mask = torch.roll(code_mask, -1, 1)  # [B, L]
        cod_labels = torch.ones(code_input.shape, dtype=torch.long, device=self.device) * -100
        cod_labels[code_mask] = code_input[code_mask]

        # calculate the loss
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), cod_labels.view(-1))

        return scores[:, 1], loss

    def finetune(self, batch, **kwargs):
        scores, cod_loss = self._get_scores(batch)
        if kwargs.get('alignment'):
            return cod_loss
        rec_loss = self.loss_fct(scores.float(), batch[Map.LBL_COL].to(self.device).float())
        return rec_loss + cod_loss

    def evaluate(self, batch):
        scores, cod_loss = self._get_scores(batch)  # [B, V=30522]
        return scores.detach().cpu().tolist()

    def get_item_alignment_tokens(self):
        prefix = self.generate_simple_input_ids('Please retrieve the item based on the following description: ')
        item = self.generate_simple_input_ids('. The item is: ')
        return prefix, item