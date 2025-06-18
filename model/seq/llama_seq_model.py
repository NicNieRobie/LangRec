from model.ctr.llama_model import LlamaModel, Llama3_1Model
from model.seq.base_seq_model import BaseSeqModel
from utils.discovery.ignore_discovery import ignore_discovery
from utils.prompts import PROMPT_SUFFIX, SIMPLE_PROMPT


@ignore_discovery
class LlamaSeqModel(BaseSeqModel, LlamaModel):
    PREFIX_PROMPT = SIMPLE_PROMPT
    SUFFIX_PROMPT = PROMPT_SUFFIX
    BIT = 16


class Llama3_1SeqModel(LlamaSeqModel):
    pass


class LlamaTulu_3_1SeqModel(LlamaSeqModel):
    pass
