import collections
import copy
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.encoding.base_encoder import BaseEncoder
from data.encoding.datasets import EmbDataset
from data.encoding.embedder import Embedder
from data.quantization.rq_vae import RQVAE
from data.quantization.rq_vae_trainer import RQVAETrainer


class SIDEncoder(BaseEncoder):
    DEFAULT_CKPT_DIR = 'ckpt/'
    DEFAULT_OUTPUT_DIR = 'encoding/'

    def __init__(self, config, device, output_dir=DEFAULT_OUTPUT_DIR, ckpt_dir=DEFAULT_CKPT_DIR):
        super().__init__(config, device, output_dir)

        self.model_name = config.model

        embedder = Embedder(self.dataset, self.model_name, self.task, config.rqvae_attrs, device)
        emb_path = embedder.run()

        self.data = EmbDataset(emb_path)

        self.num_emb_list = config.rqvae_num_emb_list
        self.sizes = f"{int(self.num_emb_list[0])}x{len(self.num_emb_list)}"

        output_name = f'{self.dataset}_{self.model_name}_{self.sizes}_{self.task}_sid.json'.lower()

        self.rqvae_ckpt_path = os.path.join(ckpt_dir, output_name)
        self.code_output_path = os.path.join(output_dir, output_name)

        self.rqvae = RQVAE(
            num_emb_list=list(map(int, config.rqvae_num_emb_list)),
            in_dim=self.data.dim,
            e_dim=config.rqvae_e_dim,
            layers=list(map(int, config.rqvae_layer_sizes)),
            dropout_prob=config.rqvae_dropout_prob,
            bn=config.rqvae_bn,
            loss_type=config.rqvae_loss_type,
            quant_loss_weight=config.rqvae_quant_loss_weight,
            beta=config.rqvae_beta,
            kmeans_init=config.rqvae_kmeans_init,
            kmeans_iters=config.rqvae_kmeans_iters,
            sk_epsilons=list(map(float, config.rqvae_sk_epsilons)),
            sk_iters=config.rqvae_sk_iters,
        )

    def encode(self) -> dict:
        best_collision_ckpt = os.path.join(self.rqvae_ckpt_path, 'best_collision_model.pth')

        if not os.path.exists(best_collision_ckpt):
            train_loader = DataLoader(self.data, num_workers=self.config.num_workers,
                                      batch_size=self.config.rqvae_batch_size, shuffle=True,
                                      pin_memory=True)

            trainer = RQVAETrainer(self.config, self.device, self.rqvae, len(train_loader), self.rqvae_ckpt_path)

            best_loss, best_collision_rate = trainer.fit(train_loader)

            print("Best RQ-VAE loss:", best_loss)
            print("Best RQ-VAE collision rate:", best_collision_rate)
        else:
            ckpt = torch.load(best_collision_ckpt, map_location='cpu', weights_only=False)
            self.rqvae.load_state_dict(ckpt["state_dict"])
            self.rqvae.to(self.device)
            self.rqvae.eval()

        enc_loader = DataLoader(self.data, batch_size=64, shuffle=False, num_workers=self.config.num_workers,
                                pin_memory=True)
        all_indices, all_indices_str, all_distances = [], [], []

        for batch in tqdm(enc_loader, desc="Encoding"):
            batch = batch.to(self.device)

            indices, distances = self.rqvae.get_indices(batch, use_sk=False)

            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            distances = distances.cpu().tolist()

            for index in indices:
                code = [int(i) for i in index]
                code_str = str(code)
                all_indices.append(code)
                all_indices_str.append(code_str)

            all_distances.extend(distances)

        all_distances = np.array(all_distances)
        sort_distances_index = np.argsort(all_distances, axis=2)
        level = len(self.num_emb_list) - 1
        max_num = int(self.num_emb_list[0])

        all_indices, all_indices_str = self.resolve_collisions(all_indices, all_indices_str, all_distances,
                                                               sort_distances_index,
                                                               level, max_num)

        print("Final collision rate:", (len(all_indices_str) - len(set(all_indices_str))) / len(all_indices_str))

        item_dict = dict(zip(range(len(self.processor.items)), self.processor.items[self.processor.ITEM_ID_COL]))

        final_dict = {
            item_dict[item_idx]: [int(code) for code in codes]
            for item_idx, codes in enumerate(all_indices)
        }

        assert len(final_dict) == len(set(str(v) for v in final_dict.values())), "Collision not resolved"

        os.makedirs(self.output_dir, exist_ok=True)
        json.dump(final_dict, open(self.code_output_path, 'w'), indent=2)

        return final_dict

    # def encode(self):
    #
    #     enc_loader = DataLoader(data, batch_size=64, shuffle=False, num_workers=config.num_workers, pin_memory=True)
    #     all_indices, all_indices_str, all_distances = [], [], []
    #
    #     for batch in tqdm(enc_loader, desc="Encoding"):
    #         batch = batch.to(device)
    #         indices, distances = model.get_indices(batch, use_sk=False)
    #         print(indices.shape)
    #         indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
    #         print('after', indices.shape)
    #         distances = distances.cpu().tolist()
    #
    #         for index in indices:
    #             code = [int(i) for i in index]
    #             code_str = str(code)
    #             all_indices.append(code)
    #             all_indices_str.append(code_str)
    #
    #         all_distances.extend(distances)
    #
    #     all_distances = np.array(all_distances)
    #     sort_distances_index = np.argsort(all_distances, axis=2)
    #     level = len(config.num_emb_list) - 1
    #     max_num = int(config.num_emb_list[0])
    #
    #     all_indices, all_indices_str = resolve_collisions(all_indices, all_indices_str, all_distances,
    #                                                       sort_distances_index,
    #                                                       level, max_num)
    #
    #     print("Total items:", len(all_indices))
    #     print("Max collisions:", max(get_indices_count(all_indices_str).values()))
    #     print("Final collision rate:", (len(all_indices_str) - len(set(all_indices_str))) / len(all_indices_str))
    #
    #     item_dict = dict(zip(range(len(processor.items)), processor.items[processor.ITEM_ID_COL]))
    #
    #     final_dict = {
    #         item_dict[item_idx]: [int(code) for code in codes]
    #         for item_idx, codes in enumerate(all_indices)
    #     }
    #
    #     assert len(final_dict) == len(set(str(v) for v in final_dict.values())), "Collision not resolved"
    #
    #     os.makedirs('./code', exist_ok=True)
    #     output_path = f'./code/{dataset}.{model_name}.{sizes}.{task}.code2'
    #     json.dump(final_dict, open(output_path, 'w'), indent=2)

    @staticmethod
    def check_collision(indices_list):
        return len(indices_list) == len(set(indices_list))

    @staticmethod
    def get_indices_count(indices_list):
        count = collections.defaultdict(int)
        for index in indices_list:
            count[index] += 1
        return count

    @staticmethod
    def get_collision_groups(indices_list):
        index_map = collections.defaultdict(list)
        for i, index in enumerate(indices_list):
            index_map[index].append(i)
        return [ids for ids in index_map.values() if len(ids) > 1]

    def resolve_collisions(self, all_indices, all_indices_str, all_distances, sort_distances_index, level, max_num):
        seen_indices = set(all_indices_str)
        item_min_dis = collections.defaultdict(list)
        for item, distances in enumerate(all_distances):
            for dis in distances:
                item_min_dis[item].append(np.min(dis))

        iteration = 0

        while True:
            if self.check_collision(all_indices_str) or iteration == 2:
                break

            collision_groups = self.get_collision_groups(all_indices_str)

            for group in collision_groups:
                min_distances = [item_min_dis[item][level] for item in group]
                sorted_items = np.argsort(min_distances)

                for i, item_idx in enumerate(sorted_items):
                    if i == 0:
                        continue

                    item = group[item_idx]
                    original_code = copy.deepcopy(all_indices[item])
                    num = i

                    while str(original_code) in seen_indices and num < max_num:
                        original_code[level] = sort_distances_index[item][level][num]
                        num += 1

                    for alt in range(1, max_num):
                        if str(original_code) not in seen_indices:
                            break

                        original_code = copy.deepcopy(all_indices[item])
                        original_code[level - 1] = sort_distances_index[item][level - 1][alt]

                        for k in range(max_num):
                            if str(original_code) not in seen_indices:
                                break

                            original_code[level] = sort_distances_index[item][level][k]

                        if str(original_code) not in seen_indices:
                            break

                    all_indices[item] = original_code
                    all_indices_str[item] = str(original_code)
                    seen_indices.add(str(original_code))

            iteration += 1

        return all_indices, all_indices_str
