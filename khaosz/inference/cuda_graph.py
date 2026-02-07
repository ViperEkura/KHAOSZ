import torch
from torch import Tensor


class CudaGraphWrapper:
    def __init__(self, function, device="cuda"):
        self.function = function
        self.device = device
        self.static_input = None
        self.static_output = None
        self.graph = None
        
    def _update_inplace(self, lhs, rhs):
        if isinstance(lhs, Tensor):
            if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype or lhs.device != rhs.device:
                raise ValueError("Tensor metadata must be static for CUDA Graph.")
            lhs.copy_(rhs)
        elif isinstance(lhs, dict):
            for k in lhs:
                self._update_inplace(lhs[k], rhs[k])
        elif isinstance(lhs, (list, tuple)):
            for i in range(len(lhs)):
                self._update_inplace(lhs[i], rhs[i])
        elif isinstance(lhs, (int, float, bool, str, type(None))):
            if lhs != rhs:
                raise ValueError("Does not support changing control parameters.")   

    def _update_kwargs(self, input_kwargs: dict):
        if self.static_input is None:
            self.static_input = input_kwargs
        else:
            self._update_inplace(self.static_input, input_kwargs)
        
    
    def run(self, input_kwargs: dict):
        self._update_kwargs(input_kwargs)
        
        if self.graph is None:
            # warmup
            _ = torch.matmul(
                torch.randn(100, 100, device=self.device),
                torch.randn(100, 100, device=self.device)
            )
            torch.cuda.synchronize()
            
            # capture graph
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_output = self.function(**self.static_input)
        
        self.graph.replay()
        
        return self.static_output