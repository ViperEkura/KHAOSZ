import torch
from torch import Tensor
from functools import wraps
from inspect import signature


class CudaGraphWrapper:
    def __init__(self, function, device="cuda", cast=False):
        self.function = function
        self.cast = cast
        self.device = device
        self.static_input = None
        self.static_output = None
        self.graph = None
        self.signature = signature(function)
        
    def _update_inplace(self, lhs, rhs):
        if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
            if lhs.shape != rhs.shape:
                raise ValueError(
                    f"Tensor shape mismatch! "
                    f"Expected: {lhs.shape}, Got: {rhs.shape}. "
                    f"Function: {self.function}"
                )
            if self.cast:
                if lhs.device != rhs.device:
                    rhs = rhs.to(device=lhs.device)
                
                if lhs.dtype != rhs.dtype:
                    rhs = rhs.to(dtype=lhs.dtype)
            else: 
                if lhs.device != rhs.device:
                    raise ValueError(
                        f"Tensor device mismatch! "
                        f"Expected: {lhs.device}, Got: {rhs.device}. "
                        f"Function: {self.function}"
                    )
                if lhs.dtype != rhs.dtype:
                    raise ValueError(
                        f"Tensor dtype mismatch! "
                        f"Expected: {lhs.dtype}, Got: {rhs.dtype}. "
                        f"Function: {self.function}"
                    )
                lhs.copy_(rhs)
        elif isinstance(lhs, dict):
            for k in lhs:
                if k in rhs:
                    self._update_inplace(lhs[k], rhs[k])
        elif isinstance(lhs, (list, tuple)):
            for i in range(len(lhs)):
                if i < len(rhs):
                    self._update_inplace(lhs[i], rhs[i])
        elif isinstance(lhs, (int, float, bool, str, type(None))):
            if lhs != rhs:
                raise ValueError("Does not support changing control parameters.")   

    def _update_args(self, input_args, input_kwargs):
        bound_args = self.signature.bind(*input_args, **input_kwargs)
        bound_args.apply_defaults()
        args_dict = bound_args.arguments
        
        if self.static_input is None:
            self.static_input = args_dict
        else:
            self._update_inplace(self.static_input, args_dict)
    
    def run(self, *args, **kwargs):
        self._update_args(args, kwargs)
        
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


def cuda_graph(device="cuda", cast=False):
    def decorator(func):
        wrapper = CudaGraphWrapper(func, device, cast)
        
        @wraps(func)
        def wrapped(*args, **kwargs):
            return wrapper.run(*args, **kwargs)
        
        return wrapped 
    
    return decorator