import torch
from dataclasses import dataclass
from torch import Tensor
from typing import List, Tuple, Union, Optional, Generator
from khaosz.inference.core import GeneratorCore, EmbeddingEncoderCore, KVCacheManager
from khaosz.config.param_config import ModelParameter


HistoryType = List[Tuple[str, str]]


def build_prompt(
    query: str,
    system_prompt: Optional[str] = None,
    history: Optional[HistoryType] = None,
) -> str:
    """
    Build prompt in ChatML format for query and history.

    Args:
        query (str): query string.
        system_prompt (Optional[str]): system prompt string.
        history (Optional[HistoryType]): history list of query and response.

    Returns:
        str: prompt string in ChatML format.
    """
    result = ""

    if system_prompt:
        result += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"

    # (convert tuple format to ChatML)
    if history:
        for user_msg, assistant_msg in history:
            result += f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            result += f"<|im_start|>assistant\n{assistant_msg}<|im_end|>\n"

    result += f"<|im_start|>user\n{query}<|im_end|>\n"
    result += "<|im_start|>assistant\n"

    return result


def pad_sequence(ids_list: List[List[int]], pad_id: int) -> Tuple[List[List[int]], int]:
    """
    Pad a list of sequences to a fixed length.

    Args:
        ids_list (List[List[int]]): A list of sequences.
        max_ids_len (int): The maximum length of sequences.
        pad_id (int): The id to pad sequences.

    Returns:
        List[List[int]]: A list of padded sequences.

    """
    max_ids_len = max(len(ids) for ids in ids_list)
    new_ids_list = []
    for ids in ids_list:
        pad_len = max_ids_len - len(ids)
        padded_seq = [pad_id] * pad_len + ids
        new_ids_list.append(padded_seq)

    return new_ids_list, max_ids_len


@dataclass
class GenerationRequest:
    """
    Request parameters for text generation.

    Attributes:
        top_k: Top-k sampling parameter.
        top_p: Top-p (nucleus) sampling parameter.
        temperature: Sampling temperature.
        max_len: Maximum generation length.
        query: Input query (string or list of strings for batch).
        history: Conversation history.
        system_prompt: System prompt for the conversation.
        stream: Whether to use streaming generation.
    """

    top_k: int
    top_p: float
    temperature: float
    max_len: int

    query: Union[str, List[str]]
    history: Optional[Union[HistoryType, List[HistoryType]]] = None
    system_prompt: Optional[str] = None
    stream: bool = False

    def __post_init__(self):
        if not isinstance(self.top_k, int) or self.top_k < 0:
            raise ValueError("top_k must be a non-negative integer")
        if not isinstance(self.top_p, float) or self.top_p < 0.0 or self.top_p > 1.0:
            raise ValueError("top_p must be a float between 0.0 and 1.0")
        if not isinstance(self.temperature, float) or self.temperature < 0.0:
            raise ValueError("temperature must be a non-negative float")


class LoopGenerator(GeneratorCore):
    def __init__(self, parameter: ModelParameter):
        super().__init__(parameter)

    def generate(self, request: GenerationRequest) -> str:
        device = next(self.model.parameters()).device
        cache_manager = KVCacheManager(self.config, 1, device=device)

        prompt = build_prompt(request.query, request.history)
        ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([ids], device=device, dtype=torch.long)

        start_cache_pos = len(ids)
        self.model.eval()
        kv_caches = cache_manager.get_kvcache()

        ids = self.generate_loop(
            input_ids,
            ids,
            request.temperature,
            request.top_k,
            request.top_p,
            kv_caches=kv_caches,
        )
        response = self.tokenizer.decode(ids[start_cache_pos:])

        return response


class StreamGenerator(GeneratorCore):
    def __init__(self, parameter: ModelParameter):
        super().__init__(parameter)

    def generate(self, request: GenerationRequest) -> Generator[str, None, None]:
        device = next(self.model.parameters()).device
        cache_manager = KVCacheManager(self.config, 1, device=device)

        prompt = build_prompt(request.query, request.history)
        ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([ids], device=device, dtype=torch.long)

        start_cache_pos = len(ids)
        cur_cache_pos = 0
        self.model.eval()
        kv_caches = cache_manager.get_kvcache()

        for _ in range(len(ids), self.config.max_len):
            next_token_id, cache_increase = self.generate_iterator(
                input_ids,
                request.temperature,
                request.top_k,
                request.top_p,
                kv_caches=kv_caches,
                start_pos=cur_cache_pos,
            )

            input_ids = next_token_id
            ids.append(next_token_id.item())
            cur_cache_pos += cache_increase

            response = self.tokenizer.decode(ids[start_cache_pos:])
            yield response

            if next_token_id.item() in self.tokenizer.stop_ids:
                yield response + "\n"
                break


class BatchGenerator(GeneratorCore):
    def __init__(self, parameter: ModelParameter):
        super().__init__(parameter)

    def generate(self, request: GenerationRequest) -> List[str]:
        batch_size = len(request.query)
        if request.history is None:
            request.history = [[] for _ in range(batch_size)]

        prompts = [
            build_prompt(query, history)
            for query, history in zip(request.query, request.history)
        ]

        ids_list = [self.tokenizer.encode(prompt) for prompt in prompts]
        ids_list, max_ids_len = pad_sequence(ids_list, self.tokenizer.pad_id)

        device = next(self.model.parameters()).device
        cache_manager = KVCacheManager(self.config, batch_size, device=device)

        input_tensor = torch.tensor(ids_list, device=device, dtype=torch.long)
        cache_manager.set_seq_mask(input_tensor, self.tokenizer.pad_id)
        activate_task_mask = [True] * batch_size

        start_cache_pos = max_ids_len
        cur_cache_pos = 0

        while max_ids_len < self.config.max_len and sum(activate_task_mask) != 0:
            kv_caches = cache_manager.get_kvcache()
            attn_mask = cache_manager.get_seq_mask()

            next_token_id, cache_increase = self.generate_iterator(
                input_tensor,
                request.temperature,
                request.top_k,
                request.top_p,
                attn_mask=attn_mask,
                kv_caches=kv_caches,
                start_pos=cur_cache_pos,
            )

            cur_cache_pos += cache_increase
            active_mask = []
            c_ids = 0

            for i in range(batch_size):
                if activate_task_mask[i]:
                    token = next_token_id[c_ids, :].item()
                    ids_list[i].append(token)
                    c_ids += 1

                    is_active = not token in self.tokenizer.stop_ids
                    activate_task_mask[i] = is_active
                    active_mask.append(is_active)

            active_mask = torch.tensor(active_mask, device=device, dtype=torch.bool)
            cache_manager.update(active_mask)
            input_tensor = next_token_id[active_mask, :]

            max_ids_len += 1

        responses = [str()] * batch_size
        for i in range(batch_size):
            responses[i] = self.tokenizer.decode(ids_list[i][start_cache_pos:])
            request.history[i].append((request.query[i], responses[i]))

        return responses


class EmbeddingEncoder(EmbeddingEncoderCore):
    def __init__(self, parameter: ModelParameter):
        super().__init__(parameter)

    def encode(self, sentence: Union[str, List[str]]) -> Union[Tensor, List[Tensor]]:
        return super().encode(sentence)


class GeneratorFactory:
    """Factory class for creating generator instances.

    Provides smart generator selection based on request characteristics:
    - Streaming: Use StreamGenerator for streaming output
    - Batch: Use BatchGenerator when query is a list
    - Single: Use LoopGenerator for single query non-streaming

    Example usage:
        generator = GeneratorFactory.create_generator(parameter, request)
        result = generator.generate(request)
    """

    @staticmethod
    def create_generator(
        parameter: ModelParameter, request: GenerationRequest
    ) -> GeneratorCore:
        """Create a generator based on request characteristics.

        Args:
            parameter: Model parameters containing model, tokenizer, config
            request: Generation request with query, options, etc.

        Returns:
            Appropriate GeneratorCore subclass instance
        """
        # Streaming generation: check stream field first
        if request.stream:
            return StreamGenerator(parameter)

        # Batch generation: query is a list of strings
        if isinstance(request.query, list):
            return BatchGenerator(parameter)

        # Default: single query non-streaming
        return LoopGenerator(parameter)

    @staticmethod
    def create_encoder(parameter: ModelParameter) -> EmbeddingEncoderCore:
        """Create an embedding encoder instance.

        Args:
            parameter: Model parameters

        Returns:
            EmbeddingEncoderCore instance
        """
        return EmbeddingEncoder(parameter)

    @classmethod
    def create(
        cls, parameter: ModelParameter, request: GenerationRequest
    ) -> GeneratorCore:
        """Convenience method that delegates to create_generator.

        Args:
            parameter: Model parameters
            request: Generation request

        Returns:
            Generator instance
        """
        return cls.create_generator(parameter, request)
