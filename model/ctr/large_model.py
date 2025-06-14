import abc

from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.prompts import PROMPT_SUFFIX, STRICT_PROMPT
from model.ctr.base_model import BaseCTRModel
from utils.auth import HF_KEY
from utils.discovery.ignore_discovery import ignore_discovery


@ignore_discovery
class LargeCTRModel(BaseCTRModel, abc.ABC):
    PREFIX_PROMPT = STRICT_PROMPT
    SUFFIX_PROMPT = PROMPT_SUFFIX
    BIT = 16

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model = AutoModelForCausalLM.from_pretrained(self.key, trust_remote_code=True, token=HF_KEY, torch_dtype=self.get_dtype())
        self.tokenizer = AutoTokenizer.from_pretrained(self.key, trust_remote_code=True, token=HF_KEY)

        self.max_len = 10_000

        self.pos_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('YES'))[0]
        self.neg_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('NO'))[0]


class Mistral7BModel(LargeCTRModel):
    NUM_LAYERS = 32

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.max_len = 1024


class Phi3_7BModel(LargeCTRModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.max_len = 2_000


class Phi2_3BModel(LargeCTRModel):
    NUM_LAYERS = 32

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.max_len = 2_000


class RecGPT7BModel(LargeCTRModel):
    NUM_LAYERS = 32

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_len = 2_000
