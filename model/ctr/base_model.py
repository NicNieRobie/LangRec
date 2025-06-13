import torch

from utils.model import match


class BaseCTRModel:
    KEY = None
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
        input_ids = self.generate_input_ids(content, wrap_prompt=True)
        input_ids = input_ids.to(self.device)

        input_len = input_ids.size(-1)

        if input_len > self.max_len:
            return None

        with torch.no_grad():
            logits = self.model(input_ids).logits

        logits = logits[0, -1, :]

        pos_score, neg_score = logits[self.pos_token].item(), logits[self.neg_token].item()
        pos_prob, _ = self.softmax(torch.tensor([pos_score, neg_score])).tolist()

        return pos_prob

    def generate_input_ids(self, content, wrap_prompt=True):
        if wrap_prompt:
            content = self.PREFIX_PROMPT + content + self.SUFFIX_PROMPT

        return self.tokenizer.encode(content, return_tensors='pt', add_special_tokens=False)

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