import abc
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.prompts import PROMPT_SUFFIX, SIMPLE_PROMPT
from model.ctr.base_model import BaseCTRModel
from utils.auth import HF_KEY
from utils.discovery.ignore_discovery import ignore_discovery


@ignore_discovery
class LlamaModel(BaseCTRModel, abc.ABC):
    PREFIX_PROMPT = SIMPLE_PROMPT
    SUFFIX_PROMPT = PROMPT_SUFFIX
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


class Llama3_1Model(LlamaModel):
    pass


class LlamaTulu_3_1Model(LlamaModel):
    pass
