from abc import ABC, abstractmethod
from tokenizers import Tokenizer, decoders, processors, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer as BpeTrainerImpl
from typing import List, Union


class BaseTokenizer(ABC):
    @abstractmethod
    def _init_tokenizer(self):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def encode(
        self,
        tokens: Union[str, List[str]],
        out_ids: bool = True,
        add_special_tokens: bool = False,
    ) -> List:
        pass

    @abstractmethod
    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @property
    @abstractmethod
    def stop_ids(self) -> List[int]:
        pass

    @property
    @abstractmethod
    def bos_id(self) -> int:
        pass

    @property
    @abstractmethod
    def eos_id(self) -> int:
        pass

    @property
    @abstractmethod
    def pad_id(self) -> int:
        pass


class BaseTrainer(ABC):
    def __init__(self, tokenizer: BaseTokenizer):
        self.tokenizer = tokenizer

    @abstractmethod
    def train(self, files, vocab_size, min_freq, **kwargs):
        pass

    @abstractmethod
    def train_from_iterator(self, iterator, vocab_size, min_freq, **kwargs):
        pass


class BpeTokenizer(BaseTokenizer):
    def __init__(
        self,
        control_tokens: List[str] = None,
        special_tokens: List[str] = None,
        path=None,
    ):
        self._control_tokens = control_tokens or [
            "<｜begin▁of▁sentence｜>",
            "<｜end▁of▁sentence｜>",
            "<｜▁pad▁｜>",
        ]
        self._special_tokens = special_tokens or [
            "<｜im▁start｜>",
            "<｜im▁end｜>",
        ]
        self._tokenizer = None
        self._init_tokenizer()
        if path is not None:
            self.load(path)

    def _init_tokenizer(self):
        model = BPE()
        self._tokenizer = Tokenizer(model)
        self._tokenizer.normalizer = normalizers.Sequence(
            [normalizers.NFC(), normalizers.Strip()]
        )
        self._tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.UnicodeScripts(),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
            ]
        )
        self._tokenizer.decoder = decoders.ByteLevel()
        self._tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    def save(self, path):
        self._tokenizer.save(path)

    def load(self, path):
        self._tokenizer = Tokenizer.from_file(path)

    def encode(
        self,
        tokens: Union[str, List[str]],
        out_ids: bool = True,
        add_special_tokens: bool = False,
    ) -> List:
        if isinstance(tokens, str):
            encoded = self._tokenizer.encode(
                tokens, add_special_tokens=add_special_tokens
            )
            return encoded.ids if out_ids else encoded.tokens
        else:
            encoded_list = self._tokenizer.encode_batch(
                tokens, add_special_tokens=add_special_tokens
            )
            return [
                encoded.ids if out_ids else encoded.tokens for encoded in encoded_list
            ]

    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def __len__(self) -> int:
        return self._tokenizer.get_vocab_size()

    @property
    def stop_ids(self) -> List[int]:
        stop_token = self._control_tokens + self._special_tokens
        return [self._tokenizer.token_to_id(tok) for tok in stop_token]

    @property
    def bos_id(self) -> int:
        return self._tokenizer.token_to_id(self._control_tokens[0])

    @property
    def eos_id(self) -> int:
        return self._tokenizer.token_to_id(self._control_tokens[1])

    @property
    def pad_id(self) -> int:
        return self._tokenizer.token_to_id(self._control_tokens[2])


class BpeTrainer(BaseTrainer):
    def __init__(self, tokenizer: BaseTokenizer):
        super().__init__(tokenizer)

    def _prepare_trainer(
        self,
        vocab_size: int,
        min_freq: int,
        reserved_token_size: int,
        max_token_length=18,
    ):
        assert reserved_token_size > len(self.tokenizer._special_tokens)
        reserved_tokens = [
            f"<|reserve{i:02d}|>"
            for i in range(reserved_token_size - len(self.tokenizer._special_tokens))
        ]
        detail_vocab_size = vocab_size - (
            len(reserved_tokens) + len(self.tokenizer._special_tokens)
        )
        alphabet = pre_tokenizers.ByteLevel.alphabet()
        min_size = len(alphabet) + len(self.tokenizer._control_tokens)
        assert detail_vocab_size > min_size

        trainer = BpeTrainerImpl(
            vocab_size=detail_vocab_size,
            min_frequency=min_freq,
            limit_alphabet=detail_vocab_size // 6,
            max_token_length=max_token_length,
            special_tokens=self.tokenizer._control_tokens,
            initial_alphabet=alphabet,
            show_progress=True,
        )
        return trainer, reserved_tokens

    def train(self, files, vocab_size, min_freq, reserved_token_size=100, **kwargs):
        trainer, reserved_tokens = self._prepare_trainer(
            vocab_size, min_freq, reserved_token_size, **kwargs
        )
        self.tokenizer._tokenizer.train(files=files, trainer=trainer)
        self.tokenizer._tokenizer.add_special_tokens(
            self.tokenizer._special_tokens + reserved_tokens
        )

    def train_from_iterator(
        self, iterator, vocab_size, min_freq, reserved_token_size=100, **kwargs
    ):
        trainer, reserved_tokens = self._prepare_trainer(
            vocab_size, min_freq, reserved_token_size, **kwargs
        )
        self.tokenizer._tokenizer.train_from_iterator(
            iterator=iterator, trainer=trainer
        )
        self.tokenizer._tokenizer.add_special_tokens(
            self.tokenizer._special_tokens + reserved_tokens
        )
