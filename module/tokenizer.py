from tokenizers import Tokenizer, Encoding
from tokenizers import decoders, processors, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


class BpeTokenizer:
    def __init__(self, path=None):
        self._special_tokens = ["<s>", "</s>", "<|user|>", "<|system|>"]
        model = BPE()
        tokenizer = Tokenizer(model)
        tokenizer.normalizer = normalizers.Sequence([
            normalizers.Strip(),
            normalizers.NFC()
        ])
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Punctuation(behavior="isolated"),
            pre_tokenizers.Whitespace(),
            pre_tokenizers.Split(pattern=r"(\d+|[a-zA-Z]+)", behavior="isolated")
        ])
        tokenizer.decoder = decoders.Sequence([decoders.ByteLevel(add_prefix_space=False, use_regex=False)])
        tokenizer.post_processor = processors.Sequence([processors.ByteLevel(trim_offsets=True)])  
        self._tokenizer = tokenizer
        
        if path is not None:
            self._tokenizer = Tokenizer.from_file(path)
    
    def __init_trainer(self, vocab_size, min_freq):
        alphabet = [chr(i) for i in range(128)]
        min_size  = len(self._special_tokens) + len(alphabet)
        assert vocab_size > min_size
        
        lim_len = vocab_size - len(self._special_tokens)
        trainer = BpeTrainer (
            initial_alphabet=alphabet,
            min_frequency=min_freq, 
            vocab_size=lim_len,
            limit_alphabet= vocab_size // 4,
            max_token_length=12,
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

    def encode(self, tokens: str, return_ids=True) -> list:
        encoded: Encoding = self._tokenizer.encode(tokens)
        return encoded.ids if return_ids else encoded.tokens

    def decode(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens)
    
    def __len__(self) -> int:
        return self._tokenizer.get_vocab_size()
    
    @property
    def stop_ids(self) -> list[int]:
        stop_ids = []
        for token in self._special_tokens:
            stop_ids.append(self._tokenizer.token_to_id(token))
        return stop_ids