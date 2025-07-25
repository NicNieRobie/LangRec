import heapq

import torch
from torch import nn

from loader.code_map import DrecCodeMap
from model.base_discrete_code_model import BaseDiscreteCodeModel
from utils.discovery.ignore_discovery import ignore_discovery


@ignore_discovery
class BaseDrecModel(BaseDiscreteCodeModel):
    PREFIX_PROMPT: str
    SUFFIX_PROMPT: str

    PREDICT_ALL = True

    def __init__(self, code_list: list[int], **kwargs):
        super().__init__(**kwargs)

        curr_idx = 0

        self.code_list = []
        self.valid_counts = []

        for num in code_list:
            self.code_list.append(slice(curr_idx, curr_idx + num))
            self.valid_counts.append(num)

            curr_idx += num

        print(self.code_list)

        self.code_tree = None
        self.code_map = None

    @classmethod
    def get_name(cls):
        return cls.__name__.replace('DrecModel', '').upper()

    def set_code_meta(self, code_tree, code_map):
        self.code_tree = code_tree
        self.code_map = code_map

    def _get_logits(self, batch):
        embeddings = self.embedding_layer(batch)
        input_embeddings = embeddings['input_embeddings']
        attention_mask = embeddings['attention_mask']

        output = self.model(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        states = output.hidden_states[-1]

        return embeddings, self.embedding_layer.classify(states)

    def _last_code_cross_entropy(self, logits, code_input, code_mask):
        code_labels_last = torch.full((code_input.size(0),), -100, dtype=torch.long, device=self.device)

        for i in range(code_input.size(0)):
            true_indices = torch.where(code_mask[i])[0]

            if true_indices.numel() > 0:
                last_idx = true_indices[-1].item()
                code_labels_last[i] = code_input[i, last_idx]

        return nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), code_labels_last)

    def _all_codes_cross_entropy(self, logits, code_input, code_mask):
        code_input = torch.roll(code_input, -1, 1)
        code_mask = torch.roll(code_mask, -1, 1)

        code_labels = torch.ones(code_input.shape, dtype=torch.long, device=self.device) * -100
        code_labels[code_mask] = code_input[code_mask]

        return nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), code_labels.view(-1))

    def finetune(self, batch, **kwargs):
        embeddings, logits = self._get_logits(batch)

        code_input = embeddings['code_input']
        code_mask = embeddings['code_mask']

        if not self.PREDICT_ALL:
            return self._last_code_cross_entropy(logits, code_input, code_mask)

        return self._all_codes_cross_entropy(logits, code_input, code_mask)

    def evaluate(self, batch):
        return self.finetune(batch)

    @staticmethod
    def _repeat_tensor(tensor, repeat_count):
        batch_size = tensor.size(0)
        repeated = tensor.repeat(repeat_count, 1)
        return repeated.view(batch_size * repeat_count, *tensor.shape[1:])

    @staticmethod
    def _filter_by_prefix(lsts, prefix):
        prefix_len = len(prefix)
        return [lst for lst in lsts if lst[:prefix_len] == prefix]

    def _get_valid_token_set(self, path, candidates):
        node = self.code_tree
        curr_idx = 0

        for idx, token in enumerate(path):
            node = node.get(token, {})
            curr_idx += 1

        candidates_by_prefix = self._filter_by_prefix(candidates, path)
        valid_tokens = [candidate[curr_idx] for candidate in candidates_by_prefix]

        return set(valid_tokens)

    def decode_prod(self, batch):
        _, logits = self._get_logits(batch) # [B, BIGL, C]
        batch_size = logits.size(0)

        decode_start = batch[DrecCodeMap.SOB_COL].to(self.device) # [B]
        decode_length = batch[DrecCodeMap.LOB_COL].to(self.device) # [B]
        if decode_length.max().item() != decode_length.min().item():
            raise ValueError('Decode length beam_length should be the same')

        # bs_ar and dl_ar are used for array comprehension: using just `:` instead creates a new array dimension
        # instead of applying the filter to the whole array
        decode_length = decode_length.max().item()
        bs_ar = torch.arange(batch_size).to(self.device)
        dl_ar = torch.arange(decode_length).to(self.device)

        ground_truth_indices = decode_start.unsqueeze(-1) + dl_ar
        input_ids = batch[DrecCodeMap.IPT_COL].to(self.device)

        # Here L is the length of the item's code
        ground_truth = input_ids[bs_ar.unsqueeze(-1), ground_truth_indices] # [B, L]
        logits = logits[bs_ar.unsqueeze(-1), ground_truth_indices] # [B, L]

        argsort = logits.argsort(dim=-1, descending=True) # [B, L, C]
        argsort = argsort.argsort(dim=-1) # [B, L, C]

        rank = argsort[bs_ar.unsqueeze(-1), dl_ar.unsqueeze(0), ground_truth] # [B, L]

        return rank

    # Taken from https://github.com/Jyonn/RecBench
    def beam_search(self, batch, search_width=3, search_mode='tree'):
        list_search = search_mode == 'easy'

        orig_batch_size = batch[DrecCodeMap.LEN_COL].size(0)
        batch[DrecCodeMap.BTH_COL] = torch.arange(orig_batch_size, device=self.device)

        beam_lengths = batch[DrecCodeMap.LOB_COL].tolist()
        max_decode_steps = int(max(beam_lengths))

        # Repeat batch
        for key in batch:
            batch[key] = self._repeat_tensor(batch[key], search_width)
        total_batch_size = orig_batch_size * search_width

        batch[DrecCodeMap.LID_COL] = torch.arange(search_width, device=self.device).repeat_interleave(orig_batch_size)

        beam_start = batch[DrecCodeMap.SOB_COL].to(self.device).unsqueeze(-1)

        # Precompute ground truth indices
        range_tensor = torch.arange(max_decode_steps, device=self.device).unsqueeze(0)
        ground_truth_indices = beam_start + range_tensor

        input_ids = batch[DrecCodeMap.IPT_COL].to(self.device)
        total_indices = torch.arange(total_batch_size, device=self.device).unsqueeze(-1)
        ground_truth = input_ids[total_indices, ground_truth_indices][:orig_batch_size]

        # Initialize beams
        last_beams = [[(0.0, [])] for _ in range(orig_batch_size)]

        for step in range(max_decode_steps):
            current_index = beam_start + step - 1

            _, logits = self._get_logits(batch)
            logits = logits[total_indices, current_index].squeeze(1)

            # Mask logits to valid tokens
            step_valid_tokens = self.code_list[step]
            mask = torch.full_like(logits, -float('inf'))
            mask[:, step_valid_tokens] = logits[:, step_valid_tokens]
            scores = torch.softmax(mask, dim=-1)

            candidate_k = search_width if list_search else self.valid_counts[step]
            topk_scores, topk_indices = scores.topk(candidate_k, dim=-1)

            new_beams = [[] for _ in range(orig_batch_size)]
            for sample_id in range(orig_batch_size):
                if step >= beam_lengths[sample_id]:
                    new_beams[sample_id] = last_beams[sample_id]
                    continue

                valid_tokens = None if list_search else self._get_valid_token_set(last_beams[sample_id][0][1], batch[DrecCodeMap.DCT_COL][sample_id])
                for beam_idx, (cur_score, cur_path) in enumerate(last_beams[sample_id]):
                    global_idx = beam_idx * orig_batch_size + sample_id

                    for token_score, token in zip(topk_scores[global_idx], topk_indices[global_idx]):
                        token = token.item()
                        if valid_tokens is not None and token not in valid_tokens:
                            continue

                        new_score = cur_score + token_score.item()
                        new_path = cur_path + [token]
                        heapq.heappush(new_beams[sample_id], (new_score, new_path))
                        if len(new_beams[sample_id]) > search_width:
                            heapq.heappop(new_beams[sample_id])

            last_beams = new_beams

            # Update input_ids with best paths
            updated_paths = []
            for sample_id in range(orig_batch_size):
                for beam_idx in range(search_width):
                    if step >= beam_lengths[sample_id]:
                        continue
                    try:
                        beam = sorted(new_beams[sample_id], key=lambda x: x[0], reverse=True)[beam_idx]
                    except IndexError:
                        beam = (0.0, [0] * (step + 1))
                    path = beam[1]
                    if step >= len(path):
                        path += [0] * (step + 1 - len(path))
                    updated_paths.append(path)
            if updated_paths:
                updated_paths_tensor = torch.tensor(updated_paths, device=self.device)
                step_range = torch.arange(step + 1, device=self.device).unsqueeze(0).expand(len(updated_paths), -1)
                replace_indices = beam_start[:len(updated_paths)] + step_range
                input_ids[total_indices[:len(updated_paths)], replace_indices] = updated_paths_tensor

        # Compute ranks
        ground_truth_str = ['-'.join(map(str, gt.tolist())) for gt in ground_truth]
        ranks = []
        for sample_id, beams in enumerate(last_beams):
            length = beam_lengths[sample_id]
            sorted_beams = sorted(beams, key=lambda x: x[0], reverse=True)
            candidates = ['-'.join(map(str, b[1][:length])) for b in sorted_beams]
            rank = next((i + 1 for i, cand in enumerate(candidates) if cand == ground_truth_str[sample_id]), -1)
            ranks.append(rank)

        return ranks

    def decode(self, batch, search_width=3, search_mode='tree'):
        if search_mode == 'prod':
            return self.decode_prod(batch)

        return self.beam_search(batch, search_width, search_mode)

    def get_special_tokens(self):
        line = self.generate_simple_input_ids('\n')
        numbers = {i: self.generate_simple_input_ids(f'({i}) ') for i in range(1, 128)}
        user = self.generate_simple_input_ids('User behavior sequence: \n')
        item = self.generate_simple_input_ids('Candidate items: ')
        prefix = self.generate_simple_input_ids(self.PREFIX_PROMPT)
        suffix = self.generate_simple_input_ids(self.SUFFIX_PROMPT)

        return line, numbers, user, item, prefix, suffix
