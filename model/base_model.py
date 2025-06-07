import torch

from utils.model import model


class BaseModel:
    KEY = None
    NUM_LAYERS: int
    PREFIX_PROMPT: str
    SUFFIX_PROMPT: str
    AS_DICT: bool = False
    BIT: int
    PEFT_TARGET_MODULES = ['q_proj', 'v_proj', 'query', 'value']

    def __init__(self, device):
        self.device = device

        self.key = model.match(self.get_name()) or self.KEY

        self.parallel = False

        self.model = None
        self.tokenizer = None

        self.max_len = None

        self.yes_token = None
        self.no_token = None

        self.loss_fct = torch.nn.BCELoss()
        self.embed_loss_fct = torch.nn.MSELoss()
        self.code_loss_fct = torch.nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax(dim=0)
        self.softmax_sft = torch.nn.Softmax()

    def load(self):
        self.model.to(self.device)
        return self

    def generate_input_ids(self, content, wrap_ask=True):
        if wrap_ask:
            content = self.PREFIX_PROMPT + content + self.SUFFIX_PROMPT

        return self.tokenizer.encode(content, return_tensors='pt', add_special_tokens=False)

    def embed(self, content, func='last', truncate=False, mask=False):
        assert func in ['last', 'pool']

        input_ids = BaseModel.generate_input_ids(self, content, wrap_ask=False)
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
