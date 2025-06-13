import torch

from model.base_model import BaseModel


class BaseCTRModel(BaseModel):
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