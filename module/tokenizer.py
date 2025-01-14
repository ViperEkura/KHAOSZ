from tokenizers import Tokenizer, Encoding
from tokenizers import decoders, processors, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


class BpeTokenizer:
    def __init__(self, path=None):
        self._special_tokens = [
            "[gMASK]", "[MASK]", "<sog>", "<eog>",
            "<|user|>", "<|system|>"
        ]
        
        model = BPE(byte_fallback=True)
        tokenizer = Tokenizer(model)
        
        tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Punctuation(), 
            pre_tokenizers.Digits(individual_digits=True)
        ])
        
        tokenizer.decoder = decoders.Sequence([
            decoders.ByteLevel(add_prefix_space=False, use_regex=True),
        ])
        tokenizer.post_processor = processors.Sequence([
            processors.ByteLevel(trim_offsets=True)
        ])  
        
        self._tokenizer = tokenizer
        
        if path is not None:
            self._tokenizer = Tokenizer.from_file(path)
    
    def __init_trainer(self, vocab_size, min_freq):
        alphabet = pre_tokenizers.ByteLevel.alphabet()
        min_size  = len(self._special_tokens) + len(alphabet)
        assert vocab_size >= min_size, f"vocab_size must be greater than {min_size}"
        
        lim_len = vocab_size - len(self._special_tokens)
        trainer = BpeTrainer (
            initial_alphabet=alphabet,
            min_frequency=min_freq, 
            vocab_size=lim_len,
            show_progress=True,
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

    def encode(self, tokens: str, out_type: int|str=int) -> list:
        assert out_type in [int, str],  f"out_type must in [int, str], but got {out_type}"
        encoded: Encoding = self._tokenizer.encode(tokens) 
        if out_type == int:
            return encoded.ids
        else:
            return encoded.tokens

    def decode(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens)
    
    def id_to_token(self, id: int) -> str:
        return self._tokenizer.id_to_token(id)
    
    def token_to_id(self, token: str) -> int:
        return self._tokenizer.token_to_id(token)
    
    def __len__(self) -> int:
        return self._tokenizer.get_vocab_size()
