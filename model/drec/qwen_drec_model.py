from model.ctr.qwen_model import QWenModel
from model.drec.base_drec_model import BaseDrecModel
from utils.discovery.ignore_discovery import ignore_discovery
from utils.prompts import DREC_SIMPLE_PROMPT, DREC_PROMPT_SUFFIX


@ignore_discovery
class QWenDrecModel(BaseDrecModel, QWenModel):
    PREFIX_PROMPT = DREC_SIMPLE_PROMPT
    SUFFIX_PROMPT = DREC_PROMPT_SUFFIX
    PEFT_TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']


class QWen2_0_5BDrecModel(QWenDrecModel):
    pass


class QWen2_1_5BDrecModel(QWenDrecModel):
    pass


class QWen2_7BDrecModel(QWenDrecModel):
    pass


class QWen2_5_7BDrecModel(QWenDrecModel):
    pass
