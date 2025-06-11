import abc

import torch
from transformers import T5Config

from model.base_model import BaseModel
from model.p5_utils.modeling import P5
from model.p5_utils.tokenization import P5Tokenizer
from utils.discovery.ignore_discovery import ignore_discovery
from utils.prompts import STRICT_PROMPT, PROMPT_SUFFIX


@ignore_discovery
class P5Model(BaseModel, abc.ABC):
    PREFIX_PROMPT = STRICT_PROMPT
    SUFFIX_PROMPT = PROMPT_SUFFIX
    BIT = 32

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.backbone, self.key = self.key['backbone'], self.key['path']

        self.max_len = 512
        config = T5Config.from_pretrained(self.backbone)
        self.start_token = config.decoder_start_token_id
        self.decoder_input_ids = torch.tensor([self.start_token]).unsqueeze(0).to(self.device)

        self.tokenizer = P5Tokenizer.from_pretrained(
            self.backbone,
            max_length=self.max_len,
            do_lower_case=False,
        )

        self.model = P5.from_pretrained(
            self.backbone,
            config=config
        )

        self.model.resize_token_embeddings(self.tokenizer.vocab_size)
        self.model.tokenizer = self.tokenizer

        self.load_state_dict()

        self.pos_token = self.tokenizer.convert_tokens_to_ids('YES')
        self.neg_token = self.tokenizer.convert_tokens_to_ids('NO')

    def load_state_dict(self):
        state_dict = torch.load(self.key, map_location='cpu')
        original_keys = list(state_dict.keys())
        for key in original_keys:
            if key.startswith("module."):
                new_key = key[len("module."):]
                state_dict[new_key] = state_dict.pop(key)
        self.model.load_state_dict(state_dict, strict=False)

    def prompt(self, content):
        input_ids = self.generate_input_ids(content, wrap_prompt=True)
        input_ids = input_ids.to(self.device)
        input_len = input_ids.size(-1)
        if input_len > self.max_len:
            return

        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                decoder_input_ids=self.decoder_input_ids,
            )
            logits = output.logits

        logits = logits[0, -1, :]
        yes_score, no_score = logits[self.pos_token].item(), logits[self.neg_token].item()

        softmax = torch.nn.Softmax(dim=0)
        yes_prob, _ = softmax(torch.tensor([yes_score, no_score])).tolist()

        return yes_prob

    def embed(self, content, func='last', truncate=False):
        input_ids = self.generate_input_ids(content, wrap_prompt=False)
        input_ids = input_ids.to(self.device)

        input_len = input_ids.size(-1)
        if input_len > self.max_len:
            return

        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                decoder_input_ids=self.decoder_input_ids,
                output_hidden_states=True
            )

        embeddings = output.decoder_last_hidden_state[0, -1, :]
        return embeddings.cpu().detach().numpy()


class P5Beauty_SmallModel(P5Model):
    pass


class P5Beauty_BaseModel(P5Model):
    pass
