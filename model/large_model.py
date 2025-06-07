import abc

from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.prompts import PROMPT_SUFFIX, STRICT_PROMPT
from model.base_model import BaseModel
from utils.auth import HF_KEY
from utils.class_library import ignore_discovery


@ignore_discovery
class LargeModel(BaseModel, abc.ABC):
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

    def _generate_input_ids(self, content):
        return self.tokenizer.encode(content, return_tensors='pt', add_special_tokens=False)


class Mistral7BModel(LargeModel):
    NUM_LAYERS = 32

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.max_len = 1024


class Phi3_7BModel(LargeModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.max_len = 2_000


class Phi2_3BModel(LargeModel):
    NUM_LAYERS = 32

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.max_len = 2_000


class RecGPT7BModel(LargeModel):
    NUM_LAYERS = 32

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_len = 2_000
