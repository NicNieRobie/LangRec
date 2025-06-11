import abc

from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.prompts import DREC_SIMPLE_PROMPT, DREC_PROMPT_SUFFIX
from model.drec.base_drec_model import BaseDrecModel
from utils.auth import HF_KEY
from utils.discovery.ignore_discovery import ignore_discovery


@ignore_discovery
class LargeDrecModel(BaseDrecModel, abc.ABC):
    PREFIX_PROMPT = DREC_SIMPLE_PROMPT
    SUFFIX_PROMPT = DREC_PROMPT_SUFFIX
    BIT = 16

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model = AutoModelForCausalLM.from_pretrained(self.key, trust_remote_code=True, token=HF_KEY, torch_dtype=self.get_dtype())
        self.tokenizer = AutoTokenizer.from_pretrained(self.key, trust_remote_code=True, token=HF_KEY)

        self.max_len = 10_000

        self.pos_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('YES'))[0]
        self.neg_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('NO'))[0]


class Mistral7BModel(LargeDrecModel):
    NUM_LAYERS = 32

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.max_len = 1024


class Phi3_7BModel(LargeDrecModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.max_len = 2_000


class Phi2_3BModel(LargeDrecModel):
    NUM_LAYERS = 32

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.max_len = 2_000


class RecGPT7BModel(LargeDrecModel):
    NUM_LAYERS = 32

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_len = 2_000
