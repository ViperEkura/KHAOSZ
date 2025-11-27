import torch
import torch.distributed as dist

from torch.utils.data import Dataset, Sampler
from typing import Optional


class ResumeableRandomSampler(Sampler[int]):
    def __init__(
        self, 
        data_source: Dataset,
        start_epoch: int=0, 
        start_iter: int=0, 
        seed: int=42,
        process_group: Optional[dist.ProcessGroup]=None,
    ):
        self.epoch = start_epoch
        self.iter = start_iter
        self.seed = seed
        self.num_samples = len(data_source)
        
        if process_group is not None:
            # input process group
            self.rank = dist.get_rank(process_group)
            self.num_replicas = dist.get_world_size(process_group)
            
        elif dist.is_available() and dist.is_initialized():
            # use default process group
            process_group = dist.group.WORLD
            self.rank = dist.get_rank()
            self.num_replicas = dist.get_world_size()
            
        else:
            # single process
            self.rank = 0
            self.num_replicas = 1

        self._indices = None
    
    def _get_indices(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.num_samples, generator=generator).tolist()
        
        self.iter = self.iter % self.num_samples
        local_indices = indices[self.rank: self.num_samples: self.num_replicas]
        self._indices = local_indices[self.iter:]
    
    def __iter__(self):
        if self._indices is None:
            self._get_indices()
        
        for i in self._indices:
            self.iter += 1
            yield i
        
        self.epoch += 1
        self._indices = None
    
    def __len__(self):
        if self._indices is None:
            self._get_indices()
        return len(self._indices)