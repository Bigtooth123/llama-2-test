from sentencepiece import SentencePieceProcessor
import os
from typing import List

class Tokenizer:
    def __init__(self, model_path: str):
        tokenizer_model_path = os.path.join(model_path, "tokenizer.model")
        if not os.path.exists(tokenizer_model_path):
            raise FileNotFoundError(f"Can not find {tokenizer_model_path}")
        self.sp_model = SentencePieceProcessor(model_file=tokenizer_model_path)

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool, truncation_len: int = int(1e9)) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            if len(t) > truncation_len-1:
                t = t[:truncation_len-1]
            t = t + [self.eos_id]
        else:
            if len(t) > truncation_len:
                t = t[:truncation_len]

        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)