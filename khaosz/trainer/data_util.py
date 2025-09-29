import torch
import bisect
import pickle as pkl
from abc import ABC, abstractmethod
from torch import Tensor
from torch.utils.data import Dataset, Sampler
from typing import Callable, List, Dict, Literal, Union

MutiSeg = Dict[str, List[Tensor]]
Seg = Dict[str, Tensor]

def load_pkl_files(paths: List[str]):
    segments: MutiSeg = {}
    total_samples = 0

    for path in paths:
        with open(path, "rb") as f:
            pkl_file: Seg = pkl.load(f)
        for key, value in pkl_file.items():
            if key not in segments:
                segments[key] = []
            segments[key].append(value)
        first_key = list(pkl_file.keys())[0]
        total_samples += pkl_file[first_key].numel()
    
    return segments, total_samples

def build_loss_mask(input_ids: Tensor, bos_token_id: int, eos_token_id: int) -> Tensor:
    token_markers = torch.zeros_like(input_ids, dtype=torch.int8)
    
    is_bos_token = input_ids.eq(bos_token_id)
    is_eos_token = input_ids.eq(eos_token_id)
    
    token_markers[is_bos_token] = 1
    token_markers[is_eos_token] = -1 
    
    cumulative_markers = torch.cumsum(token_markers, dim=-1)
    min_cumulative = cumulative_markers.min(dim=-1, keepdim=True).values
    loss_mask = cumulative_markers - min_cumulative

    return loss_mask.to(dtype=torch.bool)

def build_attention_mask(input_ids: Tensor, user_token_id: int, multi_turn: bool) -> Tensor:
    seq_len = input_ids.size(0)
    is_user_token = input_ids.eq(user_token_id)
    turn_id = is_user_token.cumsum(dim=-1)
    
    iq = turn_id.view(seq_len, 1)
    ik = turn_id.view(1, seq_len)
    
    # fix the causual attention mask
    seq_mask = (iq >= ik) if multi_turn else (iq == ik)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device)).bool()
    attention_mask = seq_mask & causal_mask
    
    return attention_mask


class BaseSegmentFetcher:
    def __init__(self, segments: List[Tensor]):
        self.segments = segments
        self.cum_lengths = []
        total = 0
        for seg in segments:
            total += len(seg)
            self.cum_lengths.append(total)
        self.total_length = total if segments else 0

    def fetch_data(self, begin_idx: int, end_idx: int) -> Tensor:
        if not (0 <= begin_idx < self.total_length and 0 <= end_idx <= self.total_length):
            raise ValueError("begin_idx or end_idx out of bounds")
        if begin_idx >= end_idx:
            return torch.tensor([], dtype=torch.long)
        
        seg_start_idx = bisect.bisect_right(self.cum_lengths, begin_idx - 1)
        seg_end_idx = bisect.bisect_left(self.cum_lengths, end_idx - 1)

        result_segments = []

        for i in range(seg_start_idx, seg_end_idx + 1):
            prev_cum = self.cum_lengths[i - 1] if i > 0 else 0
            start = max(begin_idx - prev_cum, 0)
            end = min(end_idx - prev_cum, len(self.segments[i]))
            result_segments.append(self.segments[i][start:end])

        return torch.cat(result_segments, dim=0)
    

class MutiSegmentFetcher:
    def __init__(self, muti_segments: MutiSeg):
        self.muti_keys = list(muti_segments.keys())
        self.muti_fetchers = {
            key: BaseSegmentFetcher(segments)
            for key, segments in muti_segments.items()
        }
        
    def key_fetch(self, begin_idx: int, end_idx: int, keys: Union[str, List[str]]) -> Union[Tensor, Seg]:
        fetch_dict = {} 
        keys = [keys] if isinstance(keys, str) else keys
        
        for key in keys:
            fetcher = self.muti_fetchers[key]
            fetch_tensor = fetcher.fetch_data(begin_idx, end_idx)
            fetch_dict[key] = fetch_tensor

        return fetch_dict if len(keys) > 1 else fetch_dict[keys[0]]
    
    def fetch_data(self, begin_idx: int, end_idx: int) -> Union[Tensor, Seg]:
        return self.key_fetch(begin_idx, end_idx, self.muti_keys)


class BaseDataset(Dataset, ABC):
    def __init__(self, chunk_size: int, device: str):
        super().__init__()
        self.segments: MutiSeg = {}
        self.chunk_size = chunk_size
        self.total_samples = 0
        self.device = device

    def save(self, save_path: str):      
        first_item = self.segments[keys[0]]
        segment_size = len(first_item)
        keys = list(self.segments.keys())
        
        for i in range(segment_size):
            formated_segment = {key: self.segments[key][i] for key in keys}
            pkl.dump(formated_segment, open(f"{save_path}_{i}.pkl", "wb"))
                
    
    def load(self, load_path: Union[str, List[str]]):
        paths = [load_path] if isinstance(load_path, str) else load_path
        self.segments, self.total_samples = load_pkl_files(paths)
        self.fetcher = MutiSegmentFetcher(self.segments)
        
    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        raise NotImplementedError
        
    def __len__(self) -> int:
        assert self.total_samples // self.chunk_size > 0
        return self.total_samples // self.chunk_size
    

class SeqDataset(BaseDataset):
    def __init__(
        self, 
        chunk_size, 
        device='cuda'
    ):
        super().__init__(chunk_size, device)
        self.fetcher = MutiSegmentFetcher(self.segments)

    def _fetch_data(self, begin_idx: int, end_idx: int) -> Tensor:
        return self.fetcher.key_fetch(begin_idx, end_idx, "sequence")
    
    def __getitem__(self, index):
        begin_idx = index * self.chunk_size 
        end_idx = min(begin_idx + self.chunk_size, self.total_samples - 1)
        
        x = self._fetch_data(begin_idx, end_idx).to(device=self.device, dtype=torch.long)
        y = self._fetch_data(begin_idx + 1, end_idx + 1).to(device=self.device, dtype=torch.long)
        
        return {"input_ids": x, "target_ids": y}
    
    
    
class SftDataset(BaseDataset):
    def __init__(
        self, 
        chunk_size, 
        bos_token_id, 
        eos_token_id, 
        user_token_id, 
        multi_turn=False, 
        device='cuda'
    ):
        super().__init__(chunk_size, device)
        self.fetcher = MutiSegmentFetcher(self.segments)
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.user_token_id = user_token_id
        self.multi_turn = multi_turn
    
    def _fetch_data(self, begin_idx: int, end_idx: int, key: str) -> Tensor:
        return self.fetcher.key_fetch(begin_idx, end_idx, key)
    
    def __getitem__(self, index):
        begin_idx = index * self.chunk_size 
        end_idx = min(begin_idx + self.chunk_size, self.total_samples - 1)
        
        x = self._fetch_data(begin_idx, end_idx, "sequence").to(device=self.device, dtype=torch.long)
        y = self._fetch_data(begin_idx + 1, end_idx + 1, "sequence").to(device=self.device, dtype=torch.long)
        
        loss_mask = build_loss_mask(y, self.bos_token_id, self.eos_token_id)
        attn_mask = build_attention_mask(x, self.user_token_id, self.multi_turn)
        
        return {"input_ids": x, "target_ids": y, "loss_mask": loss_mask, "attn_mask": attn_mask}


class DpoDataset(BaseDataset):
    def __init__(self, chunk_size: int, device="cuda"):
        super().__init__(chunk_size, device)
        self.fetcher = MutiSegmentFetcher(self.segments)

    def _fetch_data(self, begin_idx: int, end_idx: int, key: str) -> Tensor:
        return self.fetcher.key_fetch(begin_idx, end_idx, key)

    def __getitem__(self, index: int):
        start_idx = index * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.total_samples - 1)
        
        chosen = self._fetch_data(start_idx, end_idx, "chosen").to(device=self.device, dtype=torch.long)
        rejected = self._fetch_data(start_idx, end_idx, "rejected").to(device=self.device, dtype=torch.long)
        chosen_mask = self._fetch_data(start_idx, end_idx, "chosen_mask").to(device=self.device, dtype=torch.bool)
        rejected_mask = self._fetch_data(start_idx, end_idx, "rejected_mask").to(device=self.device, dtype=torch.bool)

        return {"chosen": chosen, "rejected": rejected, "chosen_mask": chosen_mask, "rejected_mask": rejected_mask}


class PpoDataset(BaseDataset):
    def __init__(self, chunk_size: int, device="cuda"):
        super().__init__(chunk_size, device)
        self.fetcher = MutiSegmentFetcher(self.segments)

    def _fetch_data(self, begin_idx: int, end_idx: int, key: str) -> Tensor:
        return self.fetcher.key_fetch(begin_idx, end_idx, key)
    
    def __getitem__(self, index: int) -> Dict[str, Tensor]:

        begin_idx = index * self.chunk_size
        end_idx = min(begin_idx + self.chunk_size, self.total_samples - 1)
        

        input_ids =  self._fetch_data(begin_idx, end_idx, "input_ids").to(self.device),
        actions = self._fetch_data(begin_idx, end_idx, "actions").to(self.device),
        logprobs = self._fetch_data(begin_idx, end_idx, "logprobs").to(self.device),
        rewards =  self._fetch_data(begin_idx, end_idx, "rewards").to(self.device)
        
        return {"input_ids": input_ids, "actions": actions, "logprobs": logprobs, "rewards": rewards}
    

class DatasetLoader:
    @staticmethod       
    def load(
        train_type: Literal["seq", "sft", "dpo"],
        load_path: Union[str, List[str]],
        max_len: int, 
        device: str,
        **kwargs
        ) -> BaseDataset:
        
        dataset_router: Dict[str, Callable[[int, torch.device], BaseDataset]] = {
            "seq": lambda m_len, device: SeqDataset(m_len, device=device),
            "sft": lambda m_len, device: SftDataset(
                m_len, 
                device=device,
                bos_token_id=kwargs.get("bos_token_id"),
                eos_token_id=kwargs.get("eos_token_id"),
                user_token_id=kwargs.get("user_token_id"),
                multi_turn=kwargs.get("multi_turn")
            ),
            "dpo": lambda m_len, device: DpoDataset(m_len, device=device),
        }
        dataset = dataset_router[train_type](max_len, device)
        dataset.load(load_path)
        
        return dataset
    

class RandomSampler(Sampler[int]):
    def __init__(self, data_source, generator=None, seed=42):
        self.data_source = data_source
        self.seed = seed
        self.epoch = 0
        self.current_index = 0
        self._indices = None
        
        if generator is None:
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)
        else:
            self.generator = generator
    
    def _generate_indices(self):
        n = len(self.data_source)
        self._indices = torch.randperm(n, generator=self.generator).tolist()
    
    def __iter__(self):
        n = len(self.data_source)
        
        if self._indices is None:
            self._generate_indices()
        
        for i in range(self.current_index, n):
            yield self._indices[i]
        
        self.epoch += 1
        self.current_index = 0
        self._indices = None
    
    def __len__(self):
        return len(self.data_source) - self.current_index
    
    def state_dict(self):
        return {
            'epoch': self.epoch,
            'current_index': self.current_index,
            'seed': self.seed,
            'generator_state': self.generator.get_state() if self.generator else None,
            'indices': self._indices
        }
    
    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        self.current_index = state_dict['current_index']
        self.seed = state_dict['seed']
        
        if self.generator and state_dict['generator_state'] is not None:
            self.generator.set_state(state_dict['generator_state'])
        
        self._indices = state_dict['indices']