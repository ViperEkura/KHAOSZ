"""
BPE Tokenizer Trainer module.

Provides training functionality for BPE tokenizers.
"""

from typing import List, Union

from tokenizers import pre_tokenizers
from tokenizers.trainers import BpeTrainer as BpeTrainerImpl


class BpeTrainer:
    """BPE tokenizer trainer."""

    def __init__(self, tokenizer):
        """Initialize trainer with a tokenizer instance.

        Args:
            tokenizer: A BpeTokenizer instance
        """
        self.tokenizer = tokenizer

    def _prepare_trainer(
        self,
        vocab_size: int,
        min_freq: int,
        reserved_token_size: int,
        max_token_length: int = 18,
    ):
        """Prepare the BPE trainer with proper configuration."""
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

    def train(
        self,
        files: Union[str, List[str]],
        vocab_size: int,
        min_freq: int,
        reserved_token_size: int = 100,
        **kwargs,
    ):
        """Train tokenizer from files.

        Args:
            files: Path or list of paths to training files
            vocab_size: Target vocabulary size
            min_freq: Minimum frequency for tokens
            reserved_token_size: Number of reserved tokens
            **kwargs: Additional arguments
        """
        trainer, reserved_tokens = self._prepare_trainer(
            vocab_size, min_freq, reserved_token_size, **kwargs
        )
        self.tokenizer._tokenizer.train(files=files, trainer=trainer)
        self.tokenizer._tokenizer.add_special_tokens(
            self.tokenizer._special_tokens + reserved_tokens
        )

    def train_from_iterator(
        self,
        iterator,
        vocab_size: int,
        min_freq: int,
        reserved_token_size: int = 100,
        **kwargs,
    ):
        """Train tokenizer from iterator.

        Args:
            iterator: Iterator yielding training strings
            vocab_size: Target vocabulary size
            min_freq: Minimum frequency for tokens
            reserved_token_size: Number of reserved tokens
            **kwargs: Additional arguments
        """
        trainer, reserved_tokens = self._prepare_trainer(
            vocab_size, min_freq, reserved_token_size, **kwargs
        )
        self.tokenizer._tokenizer.train_from_iterator(
            iterator=iterator, trainer=trainer
        )
        self.tokenizer._tokenizer.add_special_tokens(
            self.tokenizer._special_tokens + reserved_tokens
        )


__all__ = ["BpeTrainer"]
