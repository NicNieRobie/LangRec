import re
import torch

from peft import LoraConfig, get_peft_model

from utils.map import Map
from utils.model import match


class BaseModel:
    KEY = ""
    NUM_LAYERS: int
    PREFIX_PROMPT: str
    SUFFIX_PROMPT: str
    AS_DICT: bool = False
    BIT: int
    PEFT_TARGET_MODULES = ['q_proj', 'v_proj', 'query', 'value']

    def __init__(self, device):
        self.device = device

        self.key = match(self.get_name()) or self.KEY

        self.parallel = False

        self.model = None
        self.tokenizer = None

        self.max_len = None

        self.pos_token = None
        self.neg_token = None

        self.loss_fct = torch.nn.BCELoss()
        self.embed_loss_fct = torch.nn.MSELoss()
        self.code_loss_fct = torch.nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax(dim=0)
        self.softmax_sft = torch.nn.Softmax()

    @classmethod
    def get_name(cls):
        return cls.__name__.replace('Model', '').upper()

    def load(self):
        self.model.to(self.device)
        return self

    def __call__(self, content):
        return self.prompt(content)

    def prompt(self, content):
        pass

    def generate_input_ids(self, content, wrap_prompt=True):
        if wrap_prompt:
            content = self.PREFIX_PROMPT + content + self.SUFFIX_PROMPT

        return self.tokenizer.encode(content, return_tensors='pt', add_special_tokens=False)

    def generate_inputs(self, content, wrap_prompt=True):
        if wrap_prompt:
            content = self.PREFIX_PROMPT + content + self.SUFFIX_PROMPT

        return self.tokenizer(content, return_tensors='pt', add_special_tokens=False)

    def generate_simple_input_ids(self, content):
        return self.tokenizer.encode(content or '', add_special_tokens=False)

    def get_special_tokens(self):
        line = self.generate_simple_input_ids('\n')
        numbers = {i: self.generate_simple_input_ids(f'({i}) ') for i in range(1, 128)}
        user = self.generate_simple_input_ids('User behavior sequence: \n')
        item = self.generate_simple_input_ids('Candidate item: ')
        prefix = self.generate_simple_input_ids(self.PREFIX_PROMPT)
        suffix = self.generate_simple_input_ids(self.SUFFIX_PROMPT)

        return line, numbers, user, item, prefix, suffix

    def embed(self, content, func='last', truncate=False):
        assert func in ['last', 'pool']

        input_ids = BaseModel.generate_input_ids(self, content, wrap_prompt=False)
        input_ids = input_ids.to(self.device)

        if truncate:
            input_ids = input_ids[:, :self.max_len]

        input_len = input_ids.size(-1)

        if input_len > self.max_len:
            return

        with torch.no_grad():
            output = self.model(input_ids, output_hidden_states=True)

        if func == 'last':
            embeddings = output.hidden_states[-1][0, -1, :]
        else:
            embeddings = output.hidden_states[-1][0].mean(dim=0)

        embeddings = embeddings.float()

        return embeddings.cpu().detach().numpy()

    def get_dtype(self):
        if self.BIT == 16:
            return torch.bfloat16
        if self.BIT == 32:
            return torch.float32
        if self.BIT == 8:
            return torch.float8
        raise ValueError(f'unsupported bit: {self.BIT}')

    @property
    def embedding_dim(self):
        return self.model.config.hidden_size

    def post_init(self):
        self.model.to(self.device)
        return self

    @property
    def label_tokens(self):
        return torch.tensor([self.neg_token, self.pos_token])

    def find_lora_target_modules(self, tune_from: int, layer_prefix_pattern=r"layers?\.(\d+)"):
        """
        Automatically find modules to apply LoRA to (e.g., q_proj, v_proj, query, value),
        limited to top layers >= tune_from.
        """
        import re
        target_modules = set()

        for name, module in self.model.named_modules():
            re_match = re.search(layer_prefix_pattern, name)
            if match:
                layer_idx = int(re_match.group(1))
                if layer_idx >= tune_from:
                    if any(kw in name for kw in self.PEFT_TARGET_MODULES):
                        target_modules.add(name)

        return list(target_modules)

    def prepare_model_finetuning(self, conf, inference_mode=False, tune_from=0):
        # tune_from: 0 means finetune all layers, 1 means finetune from the first layer, etc, NUM_LAYERS means no finetuning
        # tune_from: -1 means finetune the last layer, -2 means finetune the last two layers, -NUM_LAYERS means finetune all layers
        tune_from = self.NUM_LAYERS + tune_from if tune_from < 0 else tune_from

        if not conf.use_lora:
            print(f'fully finetuning {self.get_name()} model without lora')
            return
        self.use_lora = True

        print(f'finetuning {self.get_name()} model with lora ({conf.lora_r}, {conf.lora_alpha}, {conf.lora_dropout})')
        print('tune_from:', tune_from)

        transformer_layers = []
        for name, _ in self.model.named_modules():
            if re.search(r'\.layer\.\d+|\.layers\.\d+', name):
                transformer_layers.append(name)

        # Deduplicate to find all layer blocks
        unique_layer_prefixes = sorted(set([
            re.match(r"(.*?layers?\.(\d+))", n).group(1) for n in transformer_layers if re.match(r".*layers?\.(\d+)", n)
        ]), key=lambda x: int(x.split('.')[-1]))

        # Freeze bottom layers
        for name, param in self.model.named_parameters():
            for prefix in unique_layer_prefixes:
                layer_idx = int(prefix.split('.')[-1])
                if layer_idx < tune_from and name.startswith(prefix):
                    param.requires_grad = False

        target_modules = self.find_lora_target_modules(tune_from)

        peft_config = LoraConfig(
            inference_mode=inference_mode,
            r=conf.lora_r,
            lora_alpha=conf.lora_alpha,
            lora_dropout=conf.lora_dropout,
            target_modules=target_modules,
        )
        self.model = get_peft_model(self.model, peft_config)

    def save(self, path):
        module = self.model
        if self.parallel:
            module = self.model.module
        if self.use_lora:
            # only save lora parameters
            state_dict = dict()
            for k, v in module.state_dict().items():
                if 'lora' in k:
                    state_dict[k] = v
        else:
            state_dict = module.state_dict()
        torch.save(state_dict, path)

    def load_pretrained(self, path):
        print(f'loading finetuned model from {path}')
        state_dict_ = dict()
        assert self.parallel is False  # it can be true after loading
        state_dict = torch.load(path, map_location='cpu')
        for k in state_dict:
            if k.startswith('module.'):
                state_dict_[k[7:]] = state_dict[k]
            else:
                state_dict_[k] = state_dict[k]
        self.model.load_state_dict(state_dict_, strict=False)

    def recover(self, input_ids):
        return self.tokenizer.decode(input_ids)

    def _get_scores(self, batch):
        input_ids = batch[Map.IPT_COL].to(self.device)
        length = batch[Map.LEN_COL].to(self.device)
        max_len = input_ids.size(-1)
        attention_mask = torch.arange(max_len).expand(input_ids.size(0), max_len).to(self.device) < length.view(-1, 1)
        logits = self.model(input_ids, attention_mask=attention_mask).logits  # [B, L, V]
        indices = (batch[Map.LEN_COL] - 1).view(-1, 1, 1).expand(-1, 1, logits.size(-1)).to(self.device)
        logits = torch.gather(logits, 1, indices).squeeze(1)  # [B, V]
        logits = logits[:, self.label_tokens]  # [B, 2]
        scores = self.softmax_sft(logits)  # [B, 2]
        return scores[:, 1]

    def _batch_embed(self, input_ids, length):
        input_ids = input_ids.to(self.device).detach()
        length = length.to(self.device).detach()

        max_len = input_ids.size(-1)
        attention_mask = (torch.arange(max_len, device=self.device).unsqueeze(0) < length.unsqueeze(1)).long()

        hidden_states = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[
            -1]  # [B, L, D]

        indices = torch.clamp(length - 1, min=0).view(-1, 1, 1).expand(-1, 1, hidden_states.size(-1)).to(self.device)

        embeddings = torch.gather(hidden_states, 1, indices.long()).squeeze(1)  # [B, D]
        return embeddings

    def finetune_embed(self, batch):
        user_embedding = self._batch_embed(input_ids=batch[Map.UIP_COL], length=batch[Map.UIL_COL])  # [B, D]
        item_embedding = self._batch_embed(input_ids=batch[Map.IIP_COL], length=batch[Map.IIL_COL])  # [B, D]
        similarity = torch.cosine_similarity(user_embedding, item_embedding, dim=-1)  # [B]
        # similarity = torch.einsum('bd,bd->b', user_embedding, item_embedding)  # [B]
        return self.embed_loss_fct(similarity, batch[Map.LBL_COL].to(self.device).to(self.get_dtype()))

    def evaluate_embed(self, batch):
        user_embedding = self._batch_embed(input_ids=batch[Map.UIP_COL], length=batch[Map.UIL_COL])  # [B, D]
        item_embedding = self._batch_embed(input_ids=batch[Map.IIP_COL], length=batch[Map.IIL_COL])  # [B, D]
        similarity = torch.cosine_similarity(user_embedding, item_embedding, dim=-1)  # [B]
        # similarity = torch.einsum('bd,bd->b', user_embedding, item_embedding)  # [B]
        return similarity.detach().cpu().tolist()

    def finetune(self, batch, **kwargs):
        scores = self._get_scores(batch)
        return self.loss_fct(scores.float(), batch[Map.LBL_COL].to(self.device).float())

    def evaluate(self, batch):
        scores = self._get_scores(batch)  # [B, V=30522]
        return scores.detach().cpu().tolist()