import abc

from utils.prompts import DREC_SIMPLE_PROMPT, DREC_PROMPT_SUFFIX
from model.drec.large_model import LargeDrecModel
from utils.discovery.ignore_discovery import ignore_discovery


@ignore_discovery
class QWenModel(LargeDrecModel, abc.ABC):
    PREFIX_PROMPT = DREC_SIMPLE_PROMPT
    SUFFIX_PROMPT = DREC_PROMPT_SUFFIX
    PEFT_TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']


class QWen2_7BModel(QWenModel):
    pass


class QWen2_1_5BModel(QWenModel):
    pass


class QWen2_0_5BModel(QWenModel):
    pass


class DeepSeekR1QWen_7BModel(LargeDrecModel):
    pass
