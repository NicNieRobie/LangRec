from model.ctr.llama_model import LlamaModel
from model.drec.base_drec_model import BaseDrecModel
from utils.discovery.ignore_discovery import ignore_discovery
from utils.prompts import DREC_SIMPLE_PROMPT, DREC_PROMPT_SUFFIX


@ignore_discovery
class LlamaDrecModel(BaseDrecModel, LlamaModel):
    PREFIX_PROMPT = DREC_SIMPLE_PROMPT
    SUFFIX_PROMPT = DREC_PROMPT_SUFFIX
    BIT = 16
    NUM_LAYERS = 32


class Llama3_1DrecModel(LlamaDrecModel):
    pass


class LlamaTulu_3_1DrecModel(LlamaDrecModel):
    pass
