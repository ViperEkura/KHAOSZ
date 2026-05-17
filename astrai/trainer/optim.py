import torch
from torch.optim import Optimizer


def _zeropower_via_newtonschulz(G: torch.Tensor, steps: int = 5):
    assert G.ndim == 2
    X = G.bfloat16()
    scale = max(1, G.size(0) / G.size(1)) ** 0.5
    X = X / (X.norm() + 1e-7) * scale
    if steps == 0:
        return X.type_as(G)
    a, b, c = (3.4445, -4.7750, 2.0315)
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * (A @ B)
    return X.type_as(G)


class Muon(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 2e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_lr: float = None,
        adamw_betas: tuple = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        adamw_wd: float = 0.0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_lr=adamw_lr if adamw_lr is not None else lr * 0.1,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_wd=adamw_wd,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")
                if p.ndim >= 2:
                    self._muon_update(p, grad, group)
                else:
                    self._adamw_update(p, grad, group)
        return loss

    def _muon_update(self, p, grad, group):
        lr = group["lr"]
        momentum = group["momentum"]
        wd = group["weight_decay"]
        nesterov = group["nesterov"]
        ns_steps = group["ns_steps"]
        state = self.state[p]

        p.mul_(1 - lr * wd)

        if nesterov:
            grad = grad.add(p, alpha=wd)

        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(grad)
        buf = state["momentum_buffer"]
        buf.lerp_(grad, 1 - momentum)

        update = _zeropower_via_newtonschulz(buf, steps=ns_steps)
        scale = max(1, p.size(0) / p.size(1)) ** 0.5
        p.add_(update, alpha=-lr * scale)

    def _adamw_update(self, p, grad, group):
        lr = group["adamw_lr"]
        betas = group["adamw_betas"]
        eps = group["adamw_eps"]
        wd = group["adamw_wd"]
        state = self.state[p]

        if not state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)

        state["step"] += 1
        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
        beta1, beta2 = betas

        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.lerp_(grad.square(), 1 - beta2)

        step = state["step"]
        bias1 = 1 - beta1**step
        bias2 = 1 - beta2**step

        p.mul_(1 - lr * wd)
        denom = exp_avg_sq.sqrt().div_(bias2**0.5).add_(eps)
        p.addcdiv_(exp_avg / bias1, denom, value=-lr)
