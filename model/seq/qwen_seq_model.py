from model.ctr.qwen_model import QWenModel
from model.seq.base_seq_model import BaseSeqModel
from utils.discovery.ignore_discovery import ignore_discovery
from utils.prompts import SIMPLE_PROMPT, PROMPT_SUFFIX


@ignore_discovery
class QWenSeqModel(BaseSeqModel, QWenModel):
    PREFIX_PROMPT = SIMPLE_PROMPT
    SUFFIX_PROMPT = PROMPT_SUFFIX
    BIT = 16


class QWen2_1_5BSeqModel(QWenSeqModel):
    pass


class QWen2_7BSeqModel(QWenSeqModel):
    pass


class QWen2_5_7BSeqModel(QWenSeqModel):
    pass
