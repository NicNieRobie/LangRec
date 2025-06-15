from tqdm import tqdm

from loader.code_dataset import CodeDataset
from loader.code_map import SeqCodeMap
from loader.discrete_code_preparer import DiscreteCodePreparer
from loader.token_vocab import TV
from model.seq.base_seq_model import BaseSeqModel


class SeqPreparer(DiscreteCodePreparer):
    DATASET_CLASS = CodeDataset

    model: BaseSeqModel

    def _process(self, source='finetune'):
        items = self.tokenize_items()

        line, numbers, user, item, prefix = self.model.get_special_tokens()

        datalist = []
        max_seq_len = 0

        print(f'Preprocessing {self.processor.dataset_name} dataset')

        for index, data in tqdm(
            enumerate(self.processor.generate(slicer=self.config.history_window, source=source, id_only=True)),
            total=len(self.processor.get_source_set(source=source))
        ):
            uid, history = data

            history, curr_item_id = history[:-1], history[-1]
            curr_item = items[curr_item_id]

            input_ids = prefix + user
            vocab_ids = [TV.LLM] * len(input_ids)

            beam_start = len(input_ids)
            beam_length = len(curr_item)

            for i in range(len(history)):
                input_ids += numbers[i + 1] + items[history[i]] + line
                vocab_ids += [TV.LLM] * len(numbers[i + 1]) + [TV.COD] * len(items[history[i]]) + [TV.LLM] * len(line)
                beam_start += len(numbers[i + 1]) + len(items[history[i]]) + len(line)

            input_ids += item + curr_item
            vocab_ids += [TV.LLM] * len(item) + [TV.COD] * len(curr_item)
            beam_start += len(item)

            max_seq_len = max(max_seq_len, len(input_ids))

            datalist.append({
                SeqCodeMap.IPT_COL: input_ids,
                SeqCodeMap.VOC_COL: vocab_ids,
                SeqCodeMap.SOB_COL: beam_start,
                SeqCodeMap.LOB_COL: beam_length,
                SeqCodeMap.UID_COL: uid,
                SeqCodeMap.LEN_COL: len(input_ids)
            })

        for data in datalist:
            data[SeqCodeMap.IPT_COL] = data[SeqCodeMap.IPT_COL] + [0] * (max_seq_len - data[SeqCodeMap.LEN_COL])
            data[SeqCodeMap.VOC_COL] = data[SeqCodeMap.VOC_COL] + [0] * (max_seq_len - data[SeqCodeMap.LEN_COL])
            data[SeqCodeMap.UID_COL] = self.uid_vocab.append(data[SeqCodeMap.UID_COL])

        print(f"{self.processor.dataset_name} dataset preprocessed, max seq length: {max_seq_len}")

        return datalist