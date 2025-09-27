
import torch
from abc import abstractmethod
from torch import Tensor



class MaskBuilder:
    def __init__(
        self,
        bos_token_id: int,
        eos_token_id: int,
        user_token_id: int,
        system_token_id: int,
        
    ):
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.user_token_id = user_token_id
        self.system_token_id = system_token_id
    
    @abstractmethod
    def build(input_ids: Tensor) -> Tensor:
        raise NotImplementedError



class LossMaskBuilder(MaskBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def build(self, input_ids: Tensor) -> Tensor:
        token_markers = torch.zeros_like(input_ids, dtype=torch.int8)
        
        is_user_token = input_ids.eq(self.user_token_id)
        is_system_token = input_ids.eq(self.system_token_id)
        
        token_markers[is_user_token] = 1
        token_markers[is_system_token] = -1 
        
        cumulative_markers = torch.cumsum(token_markers, dim=-1)
        min_cumulative = cumulative_markers.min(dim=-1, keepdim=True).values
        loss_mask = cumulative_markers - min_cumulative
    
        return loss_mask
        
        
        

class AttentionMaskBuilder:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def build(input_ids: Tensor):
        bsz = input_ids.size(0)