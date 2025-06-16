from tqdm import tqdm

from loader.code_dataset import CodeDataset
from loader.code_map import DrecCodeMap
from loader.discrete_code_preparer import DiscreteCodePreparer
from loader.token_vocab import TV
from model.drec.base_drec_model import BaseDrecModel


class DrecPreparer(DiscreteCodePreparer):
    DATASET_CLASS = CodeDataset

    model: BaseDrecModel

    def _process(self, source='finetune'):
        items = self.tokenize_items()

        line, numbers, user, item, prefix, suffix = self.model.get_special_tokens()

        datalist = []
        max_seq_len = 0

        print(f'Preprocessing {self.processor.dataset_name} dataset')

        for index, data in tqdm(
            enumerate(self.processor.generate(slicer=self.config.history_window, source=source, id_only=True)),
            total=len(self.processor.get_source_set(source=source))
        ):
            uid, iids, history, label = data

            history = history[-1:]
            cand_items = [items[iid] for iid in iids]

            input_ids = prefix + user
            vocab_ids = [TV.LLM] * len(input_ids)

            beam_start = len(input_ids)
            beam_length = len(label)

            for i in range(len(history)):
                input_ids += numbers[i + 1] + items[history[i]] + line
                vocab_ids += [TV.LLM] * len(numbers[i + 1]) + [TV.COD] * len(items[history[i]]) + [TV.LLM] * len(line)
                beam_start += len(numbers[i + 1]) + len(items[history[i]]) + len(line)

            input_ids += item
            vocab_ids += [TV.LLM] * len(item)

            for idx, curr_item in enumerate(cand_items):
                input_ids += numbers[idx + 1] + curr_item + line
                vocab_ids += [TV.LLM] * len(numbers[idx + 1]) + [TV.COD] * len(curr_item) + [TV.LLM] * len(line)

            beam_start += len(item)

            max_seq_len = max(max_seq_len, len(input_ids))

            datalist.append({
                DrecCodeMap.IPT_COL: input_ids,
                DrecCodeMap.VOC_COL: vocab_ids,
                DrecCodeMap.DCT_COL: cand_items,
                DrecCodeMap.SOB_COL: beam_start,
                DrecCodeMap.LOB_COL: beam_length,
                DrecCodeMap.UID_COL: uid,
                DrecCodeMap.LEN_COL: len(input_ids)
            })

        for data in datalist:
            data[DrecCodeMap.IPT_COL] = data[DrecCodeMap.IPT_COL] + [0] * (max_seq_len - data[DrecCodeMap.LEN_COL])
            data[DrecCodeMap.VOC_COL] = data[DrecCodeMap.VOC_COL] + [0] * (max_seq_len - data[DrecCodeMap.LEN_COL])
            data[DrecCodeMap.UID_COL] = self.uid_vocab.append(data[DrecCodeMap.UID_COL])

        print(f"{self.processor.dataset_name} dataset preprocessed, max seq length: {max_seq_len}")

        return datalist