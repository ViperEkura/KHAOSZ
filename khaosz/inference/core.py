import torch

from torch import Tensor 
from typing import List, Tuple, Union, Optional, Generator, Self
from khaosz.config.param_config import ModelParameter


class GeneratorCore:
    def __init__(self, parameter: ModelParameter):
        self.model = parameter.model
        self.tokenizer = parameter.tokenizer
        self.config = parameter.config

    def compute_logits(
        self,
        input_ids: Tensor,
        attn_mask: Optional[Tensor] = None,
        kv_caches: Optional[List[Tuple[Tensor, Tensor]]] = None,
        start_pos: int = 0
    ) -> Tuple[Tensor, int]:
        with torch.inference_mode():
            outputs = self.model(input_ids, attn_mask, kv_caches, start_pos)
            logits = outputs["logits"][:, -1, :]
            cache_increase = input_ids.size(-1)   
        
        return logits, cache_increase

    def to(self, *args, **kargs) -> Self:
        self.model.to(*args, **kargs)
        return self


class EmbeddingEncoderCore:
    def __init__(self, parameter: ModelParameter):
        self.model = parameter.model
        self.tokenizer = parameter.tokenizer
        self.config = parameter.config
    
    def encode(self, sentence: Union[str, List[str]]) -> Union[Tensor, List[Tensor]]:
        with_batch = isinstance(sentence, list)
        ids = self.tokenizer.encode(sentence)
        batch_ids = ids if with_batch else [ids]
        max_model_len = self.config.m_len
        
        all_fragments = []
        fragment_origin_idx = []
        
        for i, seq in enumerate(batch_ids):
            if len(seq) > max_model_len:
                fragments = [seq[j:j+max_model_len] for j in range(0, len(seq), max_model_len)]
                all_fragments.extend(fragments)
                fragment_origin_idx.extend([i] * len(fragments))
            else:
                all_fragments.append(seq)
                fragment_origin_idx.append(i)
        
        #if empty fragments
        if not all_fragments or not ids:
            return [] if with_batch else torch.tensor([])
        
        device = next(self.model.parameters()).device
        max_len = min(max(len(seq) for seq in all_fragments), max_model_len)
        
        padded_ids = []
        masks = []
        for seq in all_fragments:
            pad_len = max_len - len(seq)
            padded_seq = seq + [self.tokenizer.pad_id] * pad_len
            mask = [token_id != self.tokenizer.pad_id for token_id in padded_seq]
            padded_ids.append(padded_seq)
            masks.append(mask)
        
        input_tensor = torch.tensor(padded_ids, device=device, dtype=torch.long)
        seq_mask = torch.tensor(masks, device=device, dtype=torch.bool)
        
        with torch.inference_mode():
            outputs = self.model(input_tensor, seq_mask)["hidden_states"]
            # [num_fragments, seq_len, hidden_size]
            fragment_embs = torch.mul(outputs, seq_mask.unsqueeze(-1))  
        
        sentence_embs: List[Tensor] = []
        for i in range(len(batch_ids)):
            indices = [idx for idx, orig_idx in enumerate(fragment_origin_idx) if orig_idx == i]
            if indices is not None:
                sum_frags = torch.sum(fragment_embs[indices, :, :], dim=1)      # [frags, hidden_size]
                length = torch.sum(seq_mask[indices, :], dim=1).unsqueeze(1)    # [frags, 1]
                emb = torch.sum(sum_frags / length, dim=0)                      # [frags, hidden_size]
                sentence_embs.append(emb.flatten())
        
        if with_batch:
            return [emb.flatten() for emb in sentence_embs]
        else:
            return sentence_embs[0].flatten()

    def to(self, *args, **kargs) -> Self:
        self.model.to(*args, **kargs)
        return self