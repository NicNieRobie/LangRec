import abc

import torch

from transformers.models.bert import BertForMaskedLM
from transformers.models.bert.tokenization_bert import BertTokenizer

from utils.prompts import SIMPLE_PROMPT, PROMPT_SUFFIX
from model.base_model import BaseModel


class BertModel(BaseModel, abc.ABC):
    PREFIX_PROMPT = SIMPLE_PROMPT
    SUFFIX_PROMPT = PROMPT_SUFFIX
    BIT = 32

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model = BertForMaskedLM.from_pretrained(self.key)
        self.tokenizer = BertTokenizer.from_pretrained(self.key)

        self.max_len = self.model.config.max_position_embeddings

        self.cls_token = self.tokenizer.convert_tokens_to_ids('[CLS]')
        self.mask_token = self.tokenizer.convert_tokens_to_ids('[MASK]')

        self.pos_token = self.tokenizer.convert_tokens_to_ids('yes')
        self.neg_token = self.tokenizer.convert_tokens_to_ids('no')

    def generate_input_ids(self, content, wrap_prompt=True) -> torch.Tensor:
        if wrap_prompt:
            content = self.PREFIX_PROMPT + content + self.SUFFIX_PROMPT

        input_ids = self.tokenizer.tokenize(content)
        input_ids = self.tokenizer.convert_tokens_to_ids(input_ids)

        return torch.tensor(input_ids).unsqueeze(0)

    def get_special_tokens(self):
        line, numbers, user, item, prefix, suffix = super().get_special_tokens()
        suffix += [self.mask_token]
        return line, numbers, user, item, prefix, suffix


class BertBaseModel(BertModel):
    NUM_LAYERS = 12


class BertLargeModel(BertModel):
    NUM_LAYERS = 24
