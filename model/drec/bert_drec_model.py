from model.ctr.bert_model import BertModel
from model.drec.base_drec_model import BaseDrecModel
from utils.discovery.ignore_discovery import ignore_discovery
from utils.prompts import SIMPLE_PROMPT, PROMPT_SUFFIX


@ignore_discovery
class BertDrecModel(BaseDrecModel, BertModel):
    PREFIX_PROMPT = SIMPLE_PROMPT
    SUFFIX_PROMPT = PROMPT_SUFFIX
    BIT = 32


class BertBaseDrecModel(BertDrecModel):
    pass
