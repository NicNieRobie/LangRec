from model.ctr.bert_model import BertModel
from model.seq.base_seq_model import BaseSeqModel
from utils.discovery.ignore_discovery import ignore_discovery
from utils.prompts import SIMPLE_PROMPT, PROMPT_SUFFIX


@ignore_discovery
class BertSeqModel(BaseSeqModel, BertModel):
    PREFIX_PROMPT = SIMPLE_PROMPT
    SUFFIX_PROMPT = PROMPT_SUFFIX
    BIT = 32


class BertBaseSeqModel(BertSeqModel):
    pass
