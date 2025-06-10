import abc
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.prompts import DREC_SIMPLE_PROMPT, DREC_PROMPT_SUFFIX
from model.drec.base_drec_model import BaseDrecModel
from utils.auth import HF_KEY
from utils.discovery.ignore_discovery import ignore_discovery


@ignore_discovery
class LlamaModel(BaseDrecModel, abc.ABC):
    PREFIX_PROMPT = DREC_SIMPLE_PROMPT
    SUFFIX_PROMPT = DREC_PROMPT_SUFFIX
    BIT = 16
    NUM_LAYERS = 32

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_model_and_tokenizer()

        self.max_len = 1024

        self.pos_token = self.tokenizer.convert_tokens_to_ids('YES')
        self.neg_token = self.tokenizer.convert_tokens_to_ids('NO')

    def _load_model_and_tokenizer(self):
        load_params = {}
        if not self.key.startswith('/'):
            load_params = {
                'trust_remote_code': True,
                'token': HF_KEY,
            }

        self.model = AutoModelForCausalLM.from_pretrained(
            self.key,
            torch_dtype=self.get_dtype(),
            **load_params,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.key,
            **load_params,
        )


class Llama1Model(LlamaModel):
    pass


class Llama2Model(LlamaModel):
    pass


class Llama3Model(LlamaModel):
    pass


class Llama3_1Model(LlamaModel):
    pass
