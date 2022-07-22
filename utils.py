"""Submodule containing auxiliary functions and data structures
"""

import torch
import heapq as hp
import numpy as np
from tqdm import tqdm
from copy import deepcopy

AUTO_CLIP_TYPES = ['auto_clip', 'linear_rand_autoclip', 'quadratic_rand_autoclip']


class GradHistory:
    """Data structure with adaptive API for efficient saving gradients' lengths history
    and calculating its p-th percentile
    Args:
        clipping_type (string): type of clipping
        p_autoclip (float): p-th percentile for auto-clip clipping methods
    """

    def __init__(self, clipping_type: str, p_autoclip: float = 0.75) -> None:
        if p_autoclip <= 0 or p_autoclip > 1:  # 0 < p_autoclip ≤ 1
            raise ValueError('Invalid p_autoclip value (expected value between 0 and 1)')

        self.p_autoclip = p_autoclip
        self.clipping_type = clipping_type

        self.heap = []
        hp.heapify(self.heap)

        self.all_grad_lens = []

    def get_grad_len(self) -> float:
        """Calculates p-th percentile of all gradients' lengths
        """
        return -self.heap[0]

    def add_grad_len(self, grad_len) -> None:
        """Adds length of new gradient to the data structures
        Args:
            grad_len (float): length of new gradient
        """
        self.all_grad_lens.append(deepcopy(grad_len))

        if self.clipping_type in AUTO_CLIP_TYPES:

            if len(self.heap) / len(self.all_grad_lens) < self.p_autoclip:
                hp.heappush(self.heap, -deepcopy(grad_len))
            elif grad_len < -self.heap[0]:
                hp.heappushpop(self.heap, -deepcopy(grad_len))

    def get_len_history(self) -> list:
        """Returns whole gradients' lengths history
        """
        return self.all_grad_lens


def recursive_to(param, device):
    """Auxiliary recursive function for optimizer_to()
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
    """Function that sets up the optimizer to given device
    Args:
        optim (torch.optim.Optimizer): optimizer
        device (string): device name
    """
    for param_group in optim.param_groups:
        for param in param_group.values():
            # Not sure, there are any global tensors in the state dict
            recursive_to(param, device)


def get_optimal_L(model, optimizer, criterion, train_loader, epochs=1):
    """Function that calculates optimal Lipschitz value for SSTM optimizer
    Args:
        model (torch.nn.Module): neural network (model)
        optimizer (torch.optim.Optimizer): SSTM optimizer
        criterion (torch.nn.): criterion (loss function)
        train_loader (torch.utils.data.DataLoader): dataloader containing train data
        epochs (int, optimal): number of epochs during which Lipschitz value will be calculated
            default: 1
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
