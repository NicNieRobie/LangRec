import abc

from model.ctr.large_model import LargeCTRModel
from utils.discovery.ignore_discovery import ignore_discovery


@ignore_discovery
class QWenModel(LargeCTRModel, abc.ABC):
    PEFT_TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']


class QWen2_7BModel(QWenModel):
    pass


class QWen2_1_5BModel(QWenModel):
    pass


class QWen2_0_5BModel(QWenModel):
    pass


class DeepSeekR1QWen_7BModel(LargeCTRModel):
    pass
