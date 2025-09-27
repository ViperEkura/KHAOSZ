import re
import torch
import torch.nn.functional as F

from abc import ABC, abstractmethod
from torch import Tensor
from typing import List, Callable, Optional


class BaseTextSplitter(ABC):
    def __init__(
        self,
        max_len: int = 512,
        chunk_overlap: int = 0,
    ):
        if max_len <= 0:
            raise ValueError("max_len must be > 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")

        self.max_len = max_len
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def split(self, text: str, **kwargs) -> List[str]:
        raise NotImplementedError

    def preprocess(self, text: str) -> str:
        return text.strip()

    def postprocess(self, chunks: List[str]) -> List[str]:
        return [chunk.strip() for chunk in chunks if chunk.strip()]


class PriorityTextSplitter(BaseTextSplitter):
    def __init__(
        self,
        separators: List[str],
        max_len: int = 512,
        chunk_overlap: int = 0,
    ):
        super().__init__(max_len=max_len, chunk_overlap=chunk_overlap)
        if not separators:
            raise ValueError("separators must be a non-empty list")
        self.separators = separators

    def split(self, text: str) -> List[str]:
        text = self.preprocess(text)
        for sep in self.separators:
            parts = text.split(sep)

            valid_parts = [p.strip() for p in parts if p.strip()]
            if len(valid_parts) > 1:
                return self.postprocess(valid_parts)
        return [text]


class SemanticTextSplitter(BaseTextSplitter):

    DEFAULT_PATTERN = r'(?<=[。！？!?])(?=(?:[^"\'‘’“”]*["\'‘’“”][^"\'‘’“”]*["\'‘’“”])*[^"\'‘’“”]*$)'

    def __init__(
        self,
        embedding_func: Callable[[List[str]], List[Tensor]],
        pattern: Optional[str] = None,
        max_len: int = 512,
        chunk_overlap: int = 0,
    ):
        super().__init__(max_len=max_len, chunk_overlap=chunk_overlap)
        if not callable(embedding_func):
            raise TypeError("embedding_func must be callable")
        self.embedding_func = embedding_func
        self.pattern = pattern or SemanticTextSplitter.DEFAULT_PATTERN

    def split(
        self,
        text: str,
        threshold: float = 0.5,
        window_size: int = 1,
    ) -> List[str]:
        text = self.preprocess(text)
        sentences = [s.strip() for s in re.split(self.pattern, text) if s.strip()]

        if len(sentences) <= 1:
            return self.postprocess(sentences)

        try:
            sentence_embs = self.embedding_func(sentences)
        except Exception as e:
            raise RuntimeError(f"Embedding generation failed: {e}")

        if len(sentence_embs) != len(sentences):
            raise ValueError("Embedding function must return one vector per sentence")

        chunks = []
        emb_tensor = torch.stack(sentence_embs)  # shape: [N, D]
        current_chunk: List[str] = [sentences[0]]

        for i in range(1, len(sentences)):
            start_prev = max(0, i - window_size)
            end_prev = i
            start_next = i
            end_next = min(len(sentences), i + window_size)
            
            prev_window_emb = emb_tensor[start_prev:end_prev].mean(dim=0)
            next_window_emb = emb_tensor[start_next:end_next].mean(dim=0)

            similarity = F.cosine_similarity(
                prev_window_emb.unsqueeze(0),
                next_window_emb.unsqueeze(0),
                dim=1
            ).item()

            dynamic_threshold = max(threshold * (1 - 0.03 * (end_next - start_prev)), 0.2)

            if similarity < dynamic_threshold:
                chunks.append(" ".join(current_chunk))
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_start:]
                current_chunk.append(sentences[i])
            else:
                current_chunk.append(sentences[i])

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return self.postprocess(chunks)