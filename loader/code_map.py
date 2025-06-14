from loader.map import Map


class CodeMap(Map):
    COD_COL = 'code'
    VOC_COL = 'vocab'
    LBW_COL = 'label_weight'


class SeqCodeMap(CodeMap):
    SOB_COL = 'beam_start'
    LOB_COL = 'beam_length'
    BTH_COL = 'batch_id'
    LID_COL = 'local_id'