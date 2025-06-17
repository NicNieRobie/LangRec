from model.base_discrete_code_model import BaseDiscreteCodeModel
from model.ctr.llama_model import LlamaModel
from utils.discovery.ignore_discovery import ignore_discovery
from utils.prompts import SIMPLE_PROMPT, PROMPT_SUFFIX


@ignore_discovery
class LlamaDiscreteCodeModel(BaseDiscreteCodeModel, LlamaModel):
    PREFIX_PROMPT = SIMPLE_PROMPT
    SUFFIX_PROMPT = PROMPT_SUFFIX
    BIT = 16


class Llama3_1_DCModel(LlamaDiscreteCodeModel):
    pass
