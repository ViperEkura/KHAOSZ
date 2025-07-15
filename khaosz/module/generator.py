import torch
from torch import Tensor 
from typing import List, Tuple, Union, Optional, Generator
from .parameter import ModelParameter


def build_prompt(query, history) -> str:
    ret_prompt = ""
    if len(history) > 0:
        for his_query, his_response in history:
            ret_prompt += f"<|user|> {his_query} <|system|> <bos>{his_response}<eos>\n"
    if query is not None:
        ret_prompt += f"<|user|> {query} <|system|> <bos>"
    return ret_prompt


class GeneratorCore:
    def __init__(self, parameter: ModelParameter):
        self.model = parameter.model
        self.tokenizer = parameter.tokenizer
        self.config = parameter.config
    
    def sample_next_token(
        self, 
        ids, 
        temperature: float, 
        top_k: int, 
        top_p: float,
        attn_mask: Optional[List] = None,
        filter_value: float=-float('inf')
    ) -> Union[int, list[int]]:
        
        torch.cuda.empty_cache()
        device = next(self.model.parameters()).device
        input_tensor = torch.as_tensor(ids, device=device)
        with_batch = input_tensor.ndim == 2
        input_tensor = input_tensor.unsqueeze(0) if not with_batch else input_tensor
        attn_mask = torch.as_tensor(attn_mask, dtype=torch.bool, device=device) if attn_mask is not None else None
        
        with torch.no_grad():
            logits: Tensor = self.model(input_tensor, attn_mask)[:, -1, :]
        top_k = min(top_k, logits.size(-1)) if top_k > 0 else 0
        
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k).values[:, -1, None]
            logits[indices_to_remove] = filter_value
            
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 0] = False

            batch_indices, sorted_pos = torch.where(sorted_indices_to_remove)
            original_col_indices = sorted_indices[batch_indices, sorted_pos]
            logits[batch_indices, original_col_indices] = filter_value
        
        probabilities = torch.softmax(logits / temperature, dim=-1)
        next_token_ids = torch.multinomial(probabilities, num_samples=1)
        
        return next_token_ids.item() if not with_batch else next_token_ids.flatten().tolist()

    def to(self, *args, **kargs):
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
        if not all_fragments:
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
        
        input_tensor = torch.tensor(padded_ids, device=device)
        seq_mask = torch.tensor(masks, device=device, dtype=torch.bool)
        
        with torch.no_grad():
            outputs = self.model(input_tensor, seq_mask, return_hidden=True)
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

    def to(self, *args, **kargs):
        self.model.to(*args, **kargs)
        return self

class TextGenerator(GeneratorCore):
    def __init__(self, parameter: ModelParameter):
        super().__init__(parameter)
    
    def generate(
            self, 
            query: str, 
            temperature: float,
            top_k: int,
            top_p: float,
        ) -> str:
        
        assert temperature >= 0.0
        assert top_k >= 0
        assert top_p >= 0.0 and top_p <= 1.0
        
        
        ids = self.tokenizer.encode(query)
        start_id_pos = len(ids)
        response = str()
        
        self.model.eval()
        with torch.no_grad():
            while len(ids) < self.config.m_len:
                next_token_id = self.sample_next_token(ids, temperature, top_k=top_k, top_p=top_p)
                if next_token_id in self.tokenizer.stop_ids:
                    break
                ids.append(next_token_id)
            
        response = self.tokenizer.decode(ids[start_id_pos:])
        
        return response



class ChatGenerator(GeneratorCore):
    def __init__(self, parameter: ModelParameter):
        super().__init__(parameter)
        
    def generate(
            self, 
            query: str, 
            history: List[Tuple[str, str]],
            temperature: float,
            top_k: int,
            top_p: float,
        ) -> str:
        
        assert temperature >= 0.0
        assert top_k >= 0
        assert top_p >= 0.0 and top_p <= 1.0
        
        if history is None:
            history = []
        
        tokens = build_prompt(query, history)
        ids = self.tokenizer.encode(tokens)
        start_id_pos = len(ids)
        response = str()
        
        self.model.eval()
        with torch.no_grad():
            while len(ids) < self.config.m_len:
                next_token_id = self.sample_next_token(ids, temperature, top_k=top_k, top_p=top_p)
                if next_token_id in self.tokenizer.stop_ids:
                    break
                ids.append(next_token_id)
            
        response = self.tokenizer.decode(ids[start_id_pos:])
        history.append((query, response))
        
        return response
    

class StreamGenerator(GeneratorCore):
    def __init__(self, parameter: ModelParameter):
        super().__init__(parameter)
    
    def generate(
            self, 
            query: str, 
            history: List[Tuple[str, str]],
            temperature: float,
            top_k: int,
            top_p: float,
        ) -> Generator[Tuple[str, List[Tuple[str, str]]], None, None]:
        
        assert temperature >= 0.0
        assert top_k >= 0
        assert top_p >= 0.0 and top_p <= 1.0
        
        if history is None:
            history = list()
        
        tokens = build_prompt(query, history)
        ids = self.tokenizer.encode(tokens)
        start_id_pos = len(ids)
        response = str()
        
        self.model.eval()
        with torch.no_grad():
            while len(ids) < self.config.m_len:
                next_token_id = self.sample_next_token(ids, temperature, top_k=top_k, top_p=top_p)
                if next_token_id in self.tokenizer.stop_ids:
                    break
                ids.append(next_token_id)
                response = self.tokenizer.decode(ids[start_id_pos:])
                yield response ,history
        
        response += "\n"
        yield response, history
        
        history.append((query, response))
    

class BatchGenerator(GeneratorCore):
    def __init__(self, parameter: ModelParameter):
        super().__init__(parameter)
    
    def generate(
        self, 
        queries: List[str],
        histories: List[List[Tuple[str, str]]],
        temperature: float, 
        top_k: int, 
        top_p: float 
    ) -> List[str]:
        assert temperature >= 0.0
        assert top_k >= 0
        assert top_p >= 0.0 and top_p <= 1.0
        
        batch_size = len(queries)
        if histories is None:
            histories = [[] for _ in range(batch_size)]

        prompts = [build_prompt(query, history) for query, history in zip(queries, histories)]
        ids_list = [self.tokenizer.encode(tokens) for tokens in prompts]
        
        start_id_pos = max([len(ids) for ids in ids_list])
        padded_ids_list: List[list] = [
            [self.tokenizer.pad_id] * (start_id_pos - len(ids)) + ids 
            for ids in ids_list
        ]
        stop_flag_list = [False] * batch_size
        max_step = start_id_pos
        
        self.model.eval()
        with torch.no_grad():
            while max_step < self.config.m_len:
                active_indices = [i for i, stop in enumerate(stop_flag_list) if not stop]
                if sum(stop_flag_list) == batch_size:
                    break
                input_sequence = [padded_ids_list[i] for i in active_indices]
                attn_mask = []
                for seq in input_sequence:
                    mask = [token != self.tokenizer.pad_id for token in seq]
                    attn_mask.append(mask)
                
                next_token_ids = self.sample_next_token(
                    input_sequence, temperature, 
                    top_k=top_k, top_p=top_p,
                    attn_mask=attn_mask,
                )

                for idx, next_token_id in zip(active_indices, next_token_ids):
                    padded_ids_list[idx].append(next_token_id)
                    if next_token_id in self.tokenizer.stop_ids:
                        stop_flag_list[idx] = True
                
                max_step += 1

        responses = [""] * batch_size 
        for i in range(batch_size):
            response_ids = padded_ids_list[i][start_id_pos:]
            responses[i] = self.tokenizer.decode(response_ids)
            histories[i].append((queries[i], responses[i]))
            
        return responses


class RetrievalGenerator(GeneratorCore):
    def __init__(self, retriever_parameter: ModelParameter):
        super().__init__(retriever_parameter)
    
    def generate(
        self,
        retrieved: List[str],
        query: str, 
        history: List[Tuple[str, str]],
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> str:
        assert temperature >= 0.0
        assert top_k >= 0
        assert top_p >= 0.0 and top_p <= 1.0
        
        if history is None:
            history = []
            
        retrieved = "\n".join([f"{idx + 1}. {key}" for idx, key in enumerate(retrieved)]) if retrieved else ""
        retrieved_query = f"{retrieved}\n\n根据以上内容回答: {query}" if retrieved else query
        parameter = ModelParameter(self.model, self.tokenizer, self.config)
        
        return ChatGenerator(parameter).generate(
            retrieved_query, 
            history,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

class EmbeddingEncoder(EmbeddingEncoderCore):
    def __init__(self, parameter: ModelParameter):
        super().__init__(parameter)
        
    def encode(self, sentence: Union[str, List[str]]) -> Union[Tensor, List[Tensor]]:
        return super().encode(sentence)
        