import re

import sentencepiece as spm
from transformers import T5Tokenizer, PreTrainedTokenizer


# The special tokens of T5Tokenizer is hard-coded with <extra_id_{}>
# I create another class P5Tokenizer extending it to add <user_id_{}> & <item_id_{}>

class P5Tokenizer(T5Tokenizer):
    def __init__(
            self,
            vocab_file,
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            extra_ids=100,
            user_extra_ids=0,
            item_extra_ids=0,
            additional_special_tokens=None,
            **kwargs
    ):
        # Add extra_ids to the special token list
        if extra_ids > 0 and additional_special_tokens is None:
            additional_special_tokens = ["<extra_id_{}>".format(i) for i in range(extra_ids)]
        elif extra_ids > 0 and additional_special_tokens is not None:
            # Check that we have the right number of extra_id special tokens
            extra_tokens = len(set(filter(lambda x: bool("extra_id" in x), additional_special_tokens)))
            if extra_tokens != extra_ids:
                raise ValueError(
                    f"Both extra_ids ({extra_ids}) and additional_special_tokens ({additional_special_tokens}) are provided to T5Tokenizer. "
                    "In this case the additional_special_tokens must include the extra_ids tokens"
                )

        if user_extra_ids > 0:
            additional_special_tokens.extend(["<user_id_{}>".format(i) for i in range(user_extra_ids)])

        if item_extra_ids > 0:
            additional_special_tokens.extend(["<item_id_{}>".format(i) for i in range(item_extra_ids)])

        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)

        self.vocab_file = vocab_file
        self._extra_ids = extra_ids
        self._user_extra_ids = user_extra_ids
        self._item_extra_ids = item_extra_ids

        self.legacy = False
        self.add_prefix_space = False

        PreTrainedTokenizer.__init__(
            self,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

    @property
    def vocab_size(self):
        return self.sp_model.GetPieceSize() + self._extra_ids + self._user_extra_ids + self._item_extra_ids

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(
            i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        if token.startswith("<extra_id_"):
            match = re.match(r"<extra_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1 - self._user_extra_ids - self._item_extra_ids
        elif "<user_id_" in token:
            match = re.match(r"<user_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1 - self._item_extra_ids
        elif "<item_id_" in token:
            match = re.match(r"<item_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1
        return self.sp_model.PieceToId(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index < self.sp_model.GetPieceSize():
            token = self.sp_model.IdToPiece(index)
        else:
            if index > self.sp_model.GetPieceSize() + self._extra_ids + self._user_extra_ids - 1:
                token = "<item_id_{}>".format(self.vocab_size - 1 - index)
            elif index > self.sp_model.GetPieceSize() + self._extra_ids - 1:
                token = "<user_id_{}>".format(self.vocab_size - self._item_extra_ids - 1 - index)
            else:
                token = "<extra_id_{}>".format(
                    self.vocab_size - self._user_extra_ids - self._item_extra_ids - 1 - index)
        return token
