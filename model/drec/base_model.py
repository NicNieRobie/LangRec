from model.base_model import BaseModel


class BaseDrecModel(BaseModel):
    def prompt(self, content):
        inputs = self.generate_inputs(content, wrap_prompt=True)
        inputs = inputs.to(self.device)

        input_len = inputs["input_ids"].size(-1)

        if input_len > self.max_len:
            return None

        generated_ids = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_beams = 20,
        )

        # Return a set of tokens with the prompt itself omitted
        return self.tokenizer.decode(generated_ids[0, input_len:], skip_special_tokens=True)
