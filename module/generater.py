import os
import torch
import safetensors.torch as st

from torch import Tensor 
from typing import List, Tuple, Union, Optional
from .transformer import Config, Transformer
from .tokenizer import BpeTokenizer
from .retriever import Retriever


def build_prompt(query, history) -> str:
    ret_prompt = ""
    if len(history) > 0:
        for his_query, his_response in history:
            ret_prompt += f"<|user|> {his_query} <|system|> <bos>{his_response}<eos>\n"
    if query is not None:
        ret_prompt += f"<|user|> {query} <|system|> <bos>"
    return ret_prompt


class Khaosz:
    def __init__(self, path=None):
        self.tokenizer : BpeTokenizer = None
        self.config : Config = None
        self.model : Transformer = None
        self.retriever : Retriever = None
        
        if path is not None:
            self.load(path)
            
    def load(self, model_dir):
        tokenizer_path = os.path.join(model_dir, "tokenizer.json")
        config_path = os.path.join(model_dir, "config.json")
        weight_path = os.path.join(model_dir, "model.safetensors")
        vector_assets_path = os.path.join(model_dir, "vectorassets.json")
        
        self.tokenizer = BpeTokenizer(tokenizer_path)
        self.config = Config(config_path)
        self.model = Transformer(self.config)
        state_dict = st.load_file(weight_path)
        self.model.load_state_dict(state_dict)
        
        if os.path.exists(vector_assets_path):
            self.retriever = Retriever(vector_assets_path)
    
    def to(self, *args, **kargs):
        self.model.to(*args, **kargs)
        return self

    def sample_next_token(
        self, 
        ids, 
        temperature: float, 
        top_k: int, 
        top_p: float,
        with_batch: bool=False,
        attn_mask: Optional[Tensor] = None,
        filter_value: float=-float('inf')
    ) -> Union[int, list[int]]:
        
        torch.cuda.empty_cache()
        device = next(self.model.parameters()).device
        attn_mask = torch.as_tensor(attn_mask, dtype=torch.bool, device=device) if attn_mask is not None else None
        input_tensor = torch.as_tensor(ids, device=device)
        input_tensor = input_tensor.unsqueeze(0) if not with_batch else input_tensor
        
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
    
    
    def stream_generate(
            self, 
            query: str, 
            history: List[Tuple[str, str]]=None,
            temperature: float=1.0,
            top_k: int=0,
            top_p: float=1.0,
        ):
        
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
                next_token_id = self.sample_next_token(
                    ids, temperature, 
                    top_k=top_k, top_p=top_p
                )
                if next_token_id in self.tokenizer.stop_ids:
                    break
                ids.append(next_token_id)
                response = self.tokenizer.decode(ids[start_id_pos:])
                yield response ,history
        
        response += "\n"
        yield response, history
        
        history.append((query, response))

    def generate(
            self, 
            query: str, 
            history: List[Tuple[str, str]]=None,
            temperature: float=0.8,
            top_k: int=0,
            top_p: float=0.8,
        ):
        
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
                next_token_id = self.sample_next_token(
                    ids, temperature, 
                    top_k=top_k, top_p=top_p
                )
                if next_token_id in self.tokenizer.stop_ids:
                    break
                ids.append(next_token_id)
            
        response = self.tokenizer.decode(ids[start_id_pos:])
        history.append((query, response))
        
        return response
    
    def batch_generate(
        self, 
        queries: List[str],
        histories: List[List[Tuple[str, str]]]=None,
        temperature: float=0.95, 
        top_k: int=0, 
        top_p: float=0.8 
    ):
        assert temperature >= 0.0
        assert top_k >= 0
        assert top_p >= 0.0 and top_p <= 1.0
        
        batch_size = len(queries)
        if histories is None:
            histories = [list() for _ in range(batch_size)]

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
                    with_batch=True
                )

                for idx, next_token_id in zip(active_indices, next_token_ids):
                    padded_ids_list[idx].append(next_token_id)
                    if next_token_id in self.tokenizer.stop_ids:
                        stop_flag_list[idx] = True
                
                max_step += 1

        responses = [str()] * batch_size 
        for i in range(batch_size):
            response_ids = padded_ids_list[i][start_id_pos:]
            responses[i] = self.tokenizer.decode(response_ids)
            histories[i].append((queries[i], responses[i]))
            
        return responses
    
    def sentence_embedding(self, sentence: str) -> Tensor:
        ids = self.tokenizer.encode(sentence)
        torch.cuda.empty_cache()
        device = next(self.model.parameters()).device
        input_tensor = torch.as_tensor(ids, device=device)
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)
        
        with torch.no_grad():
            output_seg = self.model(input_tensor, return_hidden=True)
            emb_sentence = torch.squeeze(output_seg[:, -1, :], 1)
                        
        return emb_sentence
    
    def retrieve_generate(
        self,
        query: str, 
        history: List[Tuple[str, str]] = None,
        temperature: float = 0.8,
        top_k: int = 0,
        top_p: float = 0.8,
    ):
        query_tensor = self.sentence_embedding(query)
        top_k_retrieved = self.retriever.similarity(query_tensor, top_k)
        
        retrieved_context = "\n".join(
            [f"条目: {key} (得分: {score:.3f})" 
             for key, score in top_k_retrieved]
        )
        prompt = build_prompt(query, history)
        prompt_with_retrieval = f"请根据以下信息回答: {retrieved_context}\n{prompt}"
        
        ids = self.tokenizer.encode(prompt_with_retrieval)
        start_id_pos = len(ids)
        response = str()
        
        self.model.eval()
        with torch.no_grad():
            while len(ids) < self.config.m_len:
                next_token_id = self.sample_next_token(
                    ids, temperature, 
                    top_k=top_k, top_p=top_p
                )
                if next_token_id in self.tokenizer.stop_ids:
                    break
                ids.append(next_token_id)
            
        response = self.tokenizer.decode(ids[start_id_pos:])
        if history is not None:
            history.append((query, response))
        
        return response