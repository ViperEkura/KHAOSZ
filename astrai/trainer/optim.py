import torch
from torch.optim import Optimizer


def _zeropower_via_newtonschulz(G: torch.Tensor, steps: int = 5):
    assert G.ndim == 2
    X = G
    scale = max(1, G.size(0) / G.size(1)) ** 0.5
    X = X / (X.norm() + 1e-7) * scale
    if steps == 0:
        return X
    a, b, c = (3.4445, -4.7750, 2.0315)
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * (A @ B)
    return X


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
            params_2d, params_1d = [], []
            grads_2d, grads_1d = [], []

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")
                if p.ndim >= 2:
                    params_2d.append(p)
                    grads_2d.append(p.grad)
                else:
                    params_1d.append(p)
                    grads_1d.append(p.grad)

            if params_2d:
                self._muon_update_foreach(params_2d, grads_2d, group)
            if params_1d:
                self._adamw_update_foreach(params_1d, grads_1d, group)

        return loss

    def _muon_update_foreach(self, params_2d, grads_2d, group):
        lr = group["lr"]
        momentum = group["momentum"]
        wd = group["weight_decay"]
        nesterov = group["nesterov"]
        ns_steps = group["ns_steps"]

        if wd != 0:
            torch._foreach_mul_(params_2d, 1 - lr * wd)

        if nesterov:
            grads_2d = torch._foreach_add(grads_2d, params_2d, alpha=wd)

        bufs = []
        for p, grad in zip(params_2d, grads_2d):
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(grad)
            bufs.append(state["momentum_buffer"])

        torch._foreach_lerp_(bufs, grads_2d, 1 - momentum)

        for p, buf in zip(params_2d, bufs):
            update = _zeropower_via_newtonschulz(buf, steps=ns_steps)
            scale = max(1, p.size(0) / p.size(1)) ** 0.5
            p.add_(update, alpha=-lr * scale)

    def _adamw_update_foreach(self, params_1d, grads_1d, group):
        lr = group["adamw_lr"]
        betas = group["adamw_betas"]
        eps = group["adamw_eps"]
        wd = group["adamw_wd"]

        steps: list[int] = []
        exp_avgs, exp_avg_sqs = [], []
        has_state = []
        for p in params_1d:
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
                has_state.append(False)
            else:
                has_state.append(True)
            state["step"] += 1
            steps.append(state["step"])
            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])

        beta1, beta2 = betas

        torch._foreach_lerp_(exp_avgs, grads_1d, 1 - beta1)
        grads_sq = torch._foreach_mul(grads_1d, grads_1d)
        torch._foreach_lerp_(exp_avg_sqs, grads_sq, 1 - beta2)

        bias_correction1 = [1 - beta1**s for s in steps]
        bias_correction2 = [1 - beta2**s for s in steps]

        if wd != 0:
            torch._foreach_mul_(params_1d, 1 - lr * wd)

        exp_avg_corrected = torch._foreach_div(exp_avgs, bias_correction1)
        denom = torch._foreach_div(exp_avg_sqs, bias_correction2)
        denom = torch._foreach_sqrt(denom)
        torch._foreach_add_(denom, eps)
        torch._foreach_addcdiv_(params_1d, exp_avg_corrected, denom, value=-lr)
