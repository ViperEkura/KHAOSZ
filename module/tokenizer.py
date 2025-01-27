from tokenizers import Tokenizer, Encoding
from tokenizers import decoders, processors, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from typing import List, Tuple

class BpeTokenizer:
    def __init__(self, path=None):
        self._special_tokens = ["<s>", "</s>", "<|user|>", "<|system|>"]
        model = BPE()
        tokenizer = Tokenizer(model)
        tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFC()
        ])
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Punctuation(behavior="isolated"),
            pre_tokenizers.Metaspace(prepend_scheme="never"),
            pre_tokenizers.Split(pattern=r"(\d+|[a-zA-Z]+|(?:'s|'t|'re|'ve|'m|'ll|'d))", behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        ])
        tokenizer.decoder = decoders.Sequence([
            decoders.ByteLevel(),
            decoders.Metaspace(prepend_scheme="never")
        ])
        tokenizer.post_processor = processors.Sequence([
            processors.ByteLevel(trim_offsets=False)
        ])
        self._tokenizer = tokenizer
        
        if path is not None:
            self._tokenizer = Tokenizer.from_file(path)
    
    def __init_trainer(self, vocab_size, min_freq):
        alphabet = pre_tokenizers.ByteLevel.alphabet()
        min_size  = len(self._special_tokens) + len(alphabet)
        assert vocab_size > min_size
        
        lim_len = vocab_size - len(self._special_tokens)
        trainer = BpeTrainer(
            vocab_size=lim_len,
            min_frequency=min_freq, 
            limit_alphabet= vocab_size // 4,
            max_token_length=12,
            show_progress=True,
            initial_alphabet=alphabet,
        )
        return trainer
        
    def train(self, files, vocab_size, min_freq):
        trainer = self.__init_trainer(vocab_size, min_freq)
        self._tokenizer.train(files=files, trainer=trainer)
        self._tokenizer.add_special_tokens(self._special_tokens)
        
    def train_from_iterator(self, iterator, vocab_size, min_freq):
        trainer = self.__init_trainer(vocab_size, min_freq)
        self._tokenizer.train_from_iterator(iterator=iterator, trainer=trainer)
        self._tokenizer.add_special_tokens(self._special_tokens)
        
    def save(self, path):
        self._tokenizer.save(path)
        
    def load(self, path):
        self._tokenizer = Tokenizer.from_file(path)

    def encode(self, tokens: str, out_ids=True) -> List:
        encoded: Encoding = self._tokenizer.encode(tokens)
        if out_ids:
            return encoded.ids
        else:
            return encoded.tokens

    def decode(self, tokens: List[int]) -> str:
        return self._tokenizer.decode(tokens)
    
    def __len__(self) -> int:
        return self._tokenizer.get_vocab_size()
    
    @property
    def stop_ids(self) -> list[int]:
        stop_ids = []
        for token in self._special_tokens:
            stop_ids.append(self._tokenizer.token_to_id(token))
        return stop_ids