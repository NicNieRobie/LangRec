from loguru import logger
from tqdm import tqdm

from loader.code_map import CodeMap as Map
from loader.preparer import Preparer
from loader.token_vocab import TV


class CodePreparer(Preparer):
    def _process(self, source='finetune'):
        items = self.tokenize_items()

        line, numbers, user, item, prefix, suffix = self.model.get_special_tokens()

        datalist = []

        max_sequence_len = 0

        for index, data in tqdm(
            enumerate(self.processor.generate(slicer=self.config.history_window, source=source, id_only=True)),
            total=len(self.processor.get_source_set(source=source)),
            desc=f"Preprocessing the {self.processor.dataset_name} dataset"
        ):
            uid, iid, history, label = data

            current_item = items[iid][:]

            input_ids = prefix + user
            vocab_ids = [TV.LLM] * len(input_ids)

            for i in range(len(history)):
                input_ids += numbers[i + 1] + items[history[i]] + line
                vocab_ids += [TV.LLM] * len(numbers[i + 1]) + [TV.COD] * len(items[history[i]]) + [TV.LLM] * len(line)

            input_ids += item + current_item + suffix
            vocab_ids += [TV.LLM] * len(item) + [TV.COD] * len(current_item) + [TV.LLM] * len(suffix)

            max_sequence_len = max(max_sequence_len, len(input_ids))
            datalist.append({
                Map.IPT_COL: input_ids,
                Map.VOC_COL: vocab_ids,
                Map.LBL_COL: label,
                Map.UID_COL: uid,
                Map.IID_COL: iid,
                Map.LBW_COL: 1,
            })

        for data in datalist:
            data[Map.LEN_COL] = len(data[Map.IPT_COL])
            data[Map.IPT_COL] = data[Map.IPT_COL] + [0] * (max_sequence_len - data[Map.LEN_COL])
            data[Map.VOC_COL] = data[Map.VOC_COL] + [0] * (max_sequence_len - data[Map.LEN_COL])
            data[Map.UID_COL] = self.uid_vocab.append(data[Map.UID_COL])
            data[Map.IID_COL] = self.iid_vocab.append(data[Map.IID_COL])

        logger.debug(f'{self.processor.dataset_name} dataset preprocessed, max_sequence_len: {max_sequence_len}')

        return datalist