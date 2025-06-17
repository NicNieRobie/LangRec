from model.base_discrete_code_model import BaseDiscreteCodeModel
from model.ctr.bert_model import BertModel
from utils.discovery.ignore_discovery import ignore_discovery
from utils.prompts import SIMPLE_PROMPT, PROMPT_SUFFIX


@ignore_discovery
class BertDiscreteCodeModel(BaseDiscreteCodeModel, BertModel):
    PREFIX_PROMPT = SIMPLE_PROMPT
    SUFFIX_PROMPT = PROMPT_SUFFIX
    BIT = 32


class BertBase_DCModel(BertDiscreteCodeModel):
    pass
