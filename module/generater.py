import os
import torch
import safetensors.torch as st

from torch import Tensor 
from typing import List, Tuple, Union, Optional
from .transformer import Config, Transformer
from .tokenizer import BpeTokenizer
from .retriever import Retriever, TextSplitter


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
        vector_assets_path = os.path.join(model_dir, "vectors.db")
        
        self.tokenizer = BpeTokenizer(tokenizer_path)
        self.config = Config(config_path)
        self.model = Transformer(self.config)
        state_dict = st.load_file(weight_path)
        self.model.load_state_dict(state_dict)
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
    
    def sentence_embedding(self, sentence: Union[str, List[str]]) -> Union[Tensor, List[Tensor]]:
        with_batch = isinstance(sentence, list)
        encode_fn = self.tokenizer.encode
        ids = encode_fn(sentence) if not with_batch else [encode_fn(s) for s in sentence]
        
        torch.cuda.empty_cache()
        device = next(self.model.parameters()).device
        if not with_batch:
            input_tensor = torch.as_tensor(ids, device=device)
            input_tensor = input_tensor.unsqueeze(0)
            seq_mask = torch.ones_like(input_tensor, dtype=torch.bool, device=device)
        else:
            max_len = max(len(seq) for seq in ids)
            padded_ids = [[self.tokenizer.pad_id] * (max_len - len(seq)) + seq for seq in ids]
            masks = [[token != self.tokenizer.pad_id for token in seq] for seq in padded_ids]
            input_tensor = torch.as_tensor(padded_ids, device=device)
            seq_mask = torch.as_tensor(masks, device=device, dtype=torch.bool)

        with torch.no_grad():
            output_seg = self.model(input_tensor, seq_mask, return_hidden=True)
            emb_sentence = torch.mean(output_seg, dim=1)

        return emb_sentence.flatten() if not with_batch else [e.flatten() for e in emb_sentence.split(1, dim=0)]
    

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
    
    def retrieve_generate(
        self,
        query: str, 
        history: List[Tuple[str, str]] = None,
        temperature: float = 0.8,
        top_k: int = 0,
        top_p: float = 0.8,
        retrive_top_k: int = 5,
    ):
        if history is None:
            history = list()
            
        query_tensor = self.sentence_embedding(query)
        top_k_retrieved = self.retriever.retrieve(query_tensor, retrive_top_k)
        
        retrieved_context = "\n".join(
            [f"{idx + 1}. {key}" 
             for idx, (key, _) in enumerate(top_k_retrieved)]
        ) if top_k_retrieved else ""
        
        retrieved_query = f"{retrieved_context}\n\n根据以上内容回答: {query}"
        retrieved_prompt  = build_prompt(retrieved_query, history)
        
        ids = self.tokenizer.encode(retrieved_prompt)
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
    
    def chunk(self, text: str, threshold: float = 0.5, window_size: int = 1) -> List[str]:
        splitter = TextSplitter(self.sentence_embedding)
        return splitter.chunk(text, threshold, window_size)