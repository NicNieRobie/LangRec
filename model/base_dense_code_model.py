from typing import Optional, cast

import torch
from torch import nn

from loader.code_map import CodeMap as Map
from loader.token_vocab import TV
from model.base_model import BaseModel


class DenseCodeEmbeddingLayer(nn.Module):
    def __init__(
            self,
            llm_embeddings: nn.Embedding,
            device: str,
    ):
        super().__init__()

        self.llm_embeddings = llm_embeddings  # type: nn.Embedding
        # freeze llm embeddings
        self.llm_embeddings.weight.requires_grad = False

        self.cod_embeddings: Optional[nn.Embedding] = None
        self.embedding_dim = llm_embeddings.weight.shape[1]

        self.device = device

    def set_cod_embeddings(self, cod_embeddings):
        self.cod_embeddings = cod_embeddings
        self.cod_embeddings.to(self.device)
        assert self.embedding_dim == cod_embeddings.weight.shape[1]

    def get_inputs(self, batch):
        input_ids = batch[Map.IPT_COL].to(self.device)
        vocab_ids = batch[Map.VOC_COL].to(self.device)

        length = batch[Map.LEN_COL].to(self.device)
        max_len = input_ids.size(-1)
        attention_mask = torch.arange(max_len).expand(input_ids.size(0), max_len).to(self.device)
        attention_mask = cast(torch.Tensor, attention_mask < length.view(-1, 1))

        llm_mask = cast(torch.Tensor, vocab_ids == TV.LLM) & attention_mask
        cod_mask = cast(torch.Tensor, vocab_ids == TV.COD) & attention_mask
        llm_input = input_ids * llm_mask
        cod_input = input_ids * cod_mask

        return dict(
            llm_mask=llm_mask,
            cod_mask=cod_mask,
            llm_input=llm_input,
            cod_input=cod_input,
            attention_mask=attention_mask,
        )

    def forward(self, batch):
        output = self.get_inputs(batch)
        llm_input = output['llm_input']
        cod_input = output['cod_input']
        attention_mask = output['attention_mask']

        llm_embeddings = self.llm_embeddings(llm_input)
        cod_embeddings = self.cod_embeddings(cod_input)
        input_embeddings = (llm_embeddings + cod_embeddings) * attention_mask.unsqueeze(-1)
        return dict(
            **output,
            input_embeddings=input_embeddings,
        )


class BaseDenseCodeModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embedding_layer: Optional[DenseCodeEmbeddingLayer] = None
        self.embedding_dimension = self.get_token_embeddings().weight.shape[1]

        self.load_path = None

    def save(self, path):
        # torch.save(self.model.state_dict(), path)
        module = self.model
        if self.parallel:
            module = self.model.module
        if self.use_lora:
            # only save lora parameters
            state_dict = dict()
            for k, v in module.state_dict().items():
                if 'lora' in k:
                    state_dict[k] = v
        else:
            state_dict = module.state_dict()
        embedding_layer = self.embedding_layer.state_dict()
        state_dict = dict(
            model=state_dict,
            embedding_layer=embedding_layer,
        )

        torch.save(state_dict, path)

    def load_pretrained(self, path):
        self.load_path = path

        print(f'loading finetuned model from {path}')
        state_dict = torch.load(path, map_location='cpu')

        state_dict = state_dict['model']
        state_dict_ = dict()
        assert self.parallel is False  # it can be true after loading
        for k in state_dict:
            if k.startswith('module.'):
                state_dict_[k[7:]] = state_dict[k]
            else:
                state_dict_[k] = state_dict[k]
        self.model.load_state_dict(state_dict_, strict=False)

    def load_from_gist(self, path):
        print(f'loading gist model from {path}')
        state_dict = torch.load(path, map_location='cpu')
        assert self.parallel is False  # it can be true after loading

        self.model.load_state_dict(state_dict['encoder'], strict=False)

    def post_init(self):
        super().post_init()
        self.embedding_layer = DenseCodeEmbeddingLayer(
            llm_embeddings=self.get_token_embeddings(),
            device=self.device,
        )
        self.embedding_layer.to(self.device)

        if self.load_path:
            state_dict = torch.load(self.load_path, map_location='cpu')
            embedding_layer = state_dict['embedding_layer']
            self.embedding_layer.load_state_dict(embedding_layer)

    def set_cod_embeddings(self, cod_embeddings):
        return self.embedding_layer.set_cod_embeddings(cod_embeddings)

    def get_token_embeddings(self):
        return self.model.get_input_embeddings()

    def _get_scores(self, batch):
        embeddings = self.embedding_layer(batch)
        input_embeddings = embeddings['input_embeddings']
        attention_mask = embeddings['attention_mask']

        logits = self.model(inputs_embeds=input_embeddings, attention_mask=attention_mask).logits  # [B, L, V]
        indices = (batch[Map.LEN_COL] - 1).view(-1, 1, 1).expand(-1, 1, logits.size(-1)).to(self.device)
        logits = torch.gather(logits, 1, indices).squeeze(1)  # [B, V]
        logits = logits[:, self.label_tokens]  # [B, 2]
        scores = self.softmax_sft(logits)  # [B, 2]
        return scores[:, 1]