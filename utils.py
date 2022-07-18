import heapq as hp
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm


class GradLenHistory:
    """
    TODO: Class docstring
    """

    def __init__(self, clipping_type: str, p_autoclip: float = 0.75):
        if p_autoclip <= 0 or p_autoclip > 1:  # 0 < p_autoclip ≤ 1
            raise ValueError('Invalid p_autoclip value (expected value between 0 and 1)')

        self.p_autoclip = p_autoclip
        self.clipping_type = clipping_type

        self.heap = []
        hp.heapify(self.heap)

        self.all_grad_lens = []

    def get_grad_len(self):
        """
        TODO: Method docstring
        """
        return -self.heap[0]

    def add_grad_len(self, grad):
        """
        TODO: Method docstring
        """
        self.all_grad_lens.append(grad)

        if self.clipping_type in AUTO_CLIP_TYPES:

            if len(self.heap) / len(self.all_grad_lens) < self.p_autoclip:
                hp.heappush(self.heap, -grad)
            elif grad < -self.heap[0]:
                hp.heappushpop(self.heap, -grad)

    def get_history(self) -> list:
        """
        TODO: Method docstring
        """
        return self.all_grad_lens


class _RequiredParameter:
    """Singleton class representing a required parameter for an Optimizer."""

    def __repr__(self):
        return "<required parameter>"


class _DependingParameter:
    """Singleton class representing a parameter that depends on other for an Optimizer."""

    def __init__(self, other_parameter_name):
        self.other_parameter_name = other_parameter_name

    def __repr__(self):
        return f"<depends on {self.other_parameter_name}>"


def get_clipped_grad_desc_step(**kwargs):
    """
    TODO: Function docstring
    """
    type_class = {
        'no_clip': NoClip,
        'norm': NormClip,
        'layer_wise': LayerWiseClip,
        'coordinate_wise': CoordWiseClip,
        'auto_clip': AutoClip,
        'linear_stoch_autoclip': LinearStochAutoClip,
        'quadratic_stoch_autoclip': QuadraticStochAutoClip,
        'linear_stoch_norm': LinearStochNormClip,
        'quadratic_stoch_norm': QuadraticStochNormClip
    }

    try:
        clipping_class = type_class[kwargs['clipping_type']]
    except KeyError:
        raise TypeError(f'No clipping type called {kwargs["clipping_type"]}')
    else:
        return clipping_class(**kwargs)


def recursive_to(param, device):
    """
    TODO: Function docstring
    """
    # Not sure if there are any global tensors in the state dict
    if isinstance(param, torch.Tensor):
        param.data = param.data.to(device)
        if param.grad is not None:
            param.grad.data = param.grad.data.to(device)
    elif isinstance(param, dict):
        for subparam in param.values():
            recursive_to(subparam, device)
    elif isinstance(param, list):
        for subparam in param:
            recursive_to(subparam, device)


def optimizer_to(optim, device):
    """
    TODO: Function docstring
    """
    for param_group in optim.param_groups:
        for param in param_group.values():
            # Not sure, there are any global tensors in the state dict
            recursive_to(param, device)


def get_optimal_L(model, optimizer, criterion, train_loader, epochs=1):
    """
    TODO: Function docstring
    """
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model.to(device)
    optimizer_to(optimizer, device)

    g1 = []
    x1 = []
    L_history = []

    for _ in range(epochs):
        for data in tqdm(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()

            _, g2, x2 = optimizer.step()

            if all([len(g2), len(g1), len(x2), len(x1)]):
                G = 0.0
                X = 0.0
                for g_cur, g_prev, x_cur, x_prev in zip(g2, g1, x2, x1):
                    G += (g_cur - g_prev).norm() ** 2
                    X += (x_cur - x_prev).norm() ** 2
                G **= 0.5
                X **= 0.5
                L = G / X

                L_history.append(L.detach().cpu())

            optimizer.zero_grad()

            g1 = deepcopy(g2)
            x1 = deepcopy(x2)

    best_L = np.array(L_history).mean() * 2

    return best_L, L_history
