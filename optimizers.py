"""
TODO: Module docstring
"""


import heapq as hp
import random
from copy import deepcopy
from typing import List, Optional

import numpy as np
import torch
from torch import Tensor
from torch.optim import Optimizer
from tqdm import tqdm

AUTO_CLIP_TYPES = ['auto_clip', 'linear_stoch_autoclip', 'quadratic_stoch_autoclip']


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


class NoClip:
    """
    TODO: Class docstring
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        if kwargs['clipping_type'] in AUTO_CLIP_TYPES:
            self.grad_history = GradLenHistory(kwargs['clipping_type'], kwargs['p_autoclip'])
        else:
            self.grad_history = GradLenHistory(kwargs['clipping_type'])

    def get_alpha(self, **kwargs):
        """
        TODO: Method docstring
        """
        return 1

    def __call__(self,
                 params: List[Tensor],
                 d_p_list: List[Tensor],
                 momentum_buffer_list: List[Optional[Tensor]],
                 l_r: float,
                 momentum: float,
                 clipping_type: str,
                 clipping_level: float):

        grad_norm = 0.0
        for i in range(len(params)):
            grad_norm += d_p_list[i].norm() ** 2
        grad_norm = grad_norm ** 0.5

        self.grad_history.add_grad_len(grad_norm)

        for i, param in enumerate(params):
            d_p = d_p_list[i]

            if momentum != 0:
                buf = momentum_buffer_list[i]

                if buf is None:
                    buf = torch.clone(d_p).detach()
                    momentum_buffer_list[i] = buf
                else:
                    buf.mul_(momentum).add_(d_p, alpha=1)  # no dampening

                d_p = buf

            alpha = self.get_alpha(d_p=d_p, grad_norm=grad_norm, clipping_level=clipping_level,
                                   beta=self.kwargs['beta'])
            if clipping_type == 'coordinate_wise':
                param.add_(-l_r * alpha * d_p)
            else:
                param.add_(d_p, alpha=-l_r * alpha)


class NormClip(NoClip):
    """
    TODO: Class docstring
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_alpha(self, **kwargs):
        """
        TODO: Method docstring
        """
        clipping_level = kwargs['clipping_level']
        grad_norm = kwargs['grad_norm']
        return min(1, clipping_level / grad_norm)


class LinearStochNormClip(NoClip):
    """
    TODO: Class docstring
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_alpha(self, **kwargs):
        """
        TODO: Method docstring
        """
        beta = kwargs['beta']
        clipping_level = kwargs['clipping_level']
        grad_norm = kwargs['grad_norm']
        prob = pow(beta, clipping_level / grad_norm)
        clip = random.choices([True, False], weights=[prob, 1 - prob])[0]
        if not clip or clipping_level > grad_norm:
            return 1
        return min(1, clipping_level / grad_norm)


class QuadraticStochNormClip(NoClip):
    """
    TODO: Class docstring
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_alpha(self, **kwargs):
        """
        TODO: Method docstring
        """
        beta = kwargs['beta']
        clipping_level = kwargs['clipping_level']
        grad_norm = kwargs['grad_norm']
        prob = pow(beta, (clipping_level / grad_norm) ** 2)
        clip = random.choices([True, False], weights=[prob, 1 - prob])[0]
        if not clip or clipping_level > grad_norm:
            return 1
        return min(1, clipping_level / grad_norm)


class LayerWiseClip(NoClip):
    """
    TODO: Class docstring
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def get_alpha(**kwargs):
        """
        TODO: Method docstring
        """
        clipping_level = kwargs['clipping_level']
        d_p = kwargs['d_p']
        return min(1, clipping_level / d_p.norm())


class CoordWiseClip(NoClip):
    """
    TODO: Class docstring
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def get_alpha(**kwargs):
        """
        TODO: Method docstring
        """
        eps = 1e-8
        clipping_level = kwargs['clipping_level']
        d_p = kwargs['d_p']
        return torch.clip(clipping_level / (torch.abs(d_p) + eps), min=0, max=1)


class AutoClip(NoClip):
    """
    TODO: Class docstring
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_alpha(self, **kwargs):
        """
        TODO: Method docstring
        """
        grad_norm_p = self.grad_history.get_grad_len()
        grad_norm = kwargs['grad_norm']
        return min(1, grad_norm_p / grad_norm)


class LinearStochAutoClip(NoClip):
    """
    TODO: Class docstring
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_alpha(self, **kwargs):
        """
        TODO: Method docstring
        """
        beta = kwargs['beta']
        grad_norm_p = self.grad_history.get_grad_len()
        grad_norm = kwargs['grad_norm']
        prob = pow(beta, grad_norm_p / grad_norm)
        clip = random.choices([True, False], weights=[prob, 1 - prob])[0]
        if not clip or grad_norm_p > grad_norm:
            return 1
        return min(1, grad_norm_p / grad_norm)


class QuadraticStochAutoClip(NoClip):
    """
    TODO: Class docstring
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_alpha(self, **kwargs):
        """
        TODO: Method docstring
        """
        beta = kwargs['beta']
        grad_norm_p = self.grad_history.get_grad_len()
        grad_norm = kwargs['grad_norm']
        prob = pow(beta, (grad_norm_p / grad_norm) ** 2)
        clip = random.choices([True, False], weights=[prob, 1 - prob])[0]
        if not clip or grad_norm_p > grad_norm:
            return 1
        return min(1, grad_norm_p / grad_norm)


def get_clipped_grad_desc_step(**kwargs):
    """
    TODO: Function docstring
    """
    if kwargs['clipping_type'] == 'no_clip':
        return NoClip(**kwargs)
    if kwargs['clipping_type'] == 'norm':
        return NormClip(**kwargs)
    if kwargs['clipping_type'] == 'layer_wise':
        return LayerWiseClip(**kwargs)
    if kwargs['clipping_type'] == 'coordinate_wise':
        return CoordWiseClip(**kwargs)
    if kwargs['clipping_type'] == 'auto_clip':
        return AutoClip(**kwargs)
    if kwargs['clipping_type'] == 'linear_stoch_autoclip':
        return LinearStochAutoClip(**kwargs)
    if kwargs['clipping_type'] == 'quadratic_stoch_autoclip':
        return QuadraticStochAutoClip(**kwargs)
    if kwargs['clipping_type'] == 'linear_stoch_norm':
        return LinearStochNormClip(**kwargs)
    if kwargs['clipping_type'] == 'quadratic_stoch_norm':
        return QuadraticStochNormClip(**kwargs)
    raise TypeError(f'No clipping type called {kwargs["clipping_type"]}')


class _RequiredParameter:
    """Singleton class representing a required parameter for an Optimizer."""

    def __repr__(self):
        return "<required parameter>"


required = _RequiredParameter()


class _DependingParameter:
    """Singleton class representing a parameter that depends on other for an Optimizer."""

    def __init__(self, other_parameter_name):
        self.other_parameter_name = other_parameter_name

    def __repr__(self):
        return f"<depends on {self.other_parameter_name}>"


depending = _DependingParameter()


class ClippedSGD(Optimizer):
    r"""Implements clipped version of stochastic gradient descent
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        l_r (float): learning rate
        clipping_type (string, optional):
        type of clipping to use: 'norm'|'layer_wise'|'coordinate_wise'.
            'no_clip': no clipping, standart sgd;
            'norm': standard clipping, \min\{1,\lambda / \|\nabla f(x)\|\} \nabla f(x);
            'layer_wise': standard clipping,
                but it is calculated for each layer independently;
            'coordinate_wise': coordinate wise clipping,
                \min\{1_n, \lambda / \nabla f(x)\} \nabla f(x)
            where all operations are coordinate wise. (Default: 'norm')
        clipping_level (float, optional):
        level of clipping \lambda (see clipping_type for more information).
            Default value depends on clipping_type:
            for clipping_type='norm' default clipping_level=1
            for clipping_type='layer_wise' default clipping_level=1
            for clipping_type='coordinate_wise' default clipping_level=0.1
    Example:
        >>> optimizer = torch.optim.ClippedSGD(model.parameters(), l_r=0.01,
                                                clipping_type='layer_wise', clipping_level=10)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(
            self, params,
            l_r=required,
            momentum=0,
            clipping_type='no_clip',
            clipping_level=_DependingParameter('clipping_type'),
            beta=0,
            **kwargs
    ):
        if l_r is not required and l_r < 0.0:
            raise ValueError("Invalid learning rate: {}".format(l_r))
        type_to_default_level = {
            'no_clip': 0.0,
            'norm': 1.0,
            'layer_wise': 0.3,
            'coordinate_wise': 0.1,
            'auto_clip': 1.0,
            'linear_stoch_norm': 1.0,
            'quadratic_stoch_norm': 1.0,
            'linear_stoch_autoclip': 1.0,
            'quadratic_stoch_autoclip': 1.0
        }
        if clipping_type not in type_to_default_level:
            raise ValueError("Invalid clipping type: {}, possible types are {}".
                             format(l_r, list(type_to_default_level.keys())))
        if not isinstance(clipping_level, depending) and clipping_level < 0.0:
            raise ValueError("Invalid clipping level: {}".format(clipping_level))
        if isinstance(clipping_level, depending):
            clipping_level = type_to_default_level[clipping_type]
        defaults = dict(
            l_r=l_r,
            momentum=momentum,
            clipping_type=clipping_type, clipping_level=clipping_level
        )

        kwargs['clipping_type'] = clipping_type
        kwargs['clipping_level'] = clipping_level
        kwargs['beta'] = beta

        self.grad_desc_step = get_clipped_grad_desc_step(**kwargs)
        super(ClippedSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(ClippedSGD, self).__setstate__(state)

    @torch.no_grad()  # sets all requires_grad flags to False
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            l_r = group['l_r']
            momentum = group['momentum']
            clipping_type = group['clipping_type']
            clipping_level = group['clipping_level']

            for p in group['params']:
                if p.grad is not None:
                    d_p_list.append(p.grad)
                    params_with_grad.append(p)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            # update parameters
            self.grad_desc_step(
                params_with_grad,
                d_p_list,
                momentum_buffer_list,
                l_r,
                momentum,
                clipping_type,
                clipping_level
            )

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss


class ClippedSSTM(Optimizer):
    r"""Implements Clipped Stochastic Similar Triangles Method
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        l_r (float): learning rate (stepsize parameter an in paper, inverse to real l_r)
        L (float): Lipschitz constant
        clipping_type (string, optional):
        type of clipping to use: 'norm'|'layer_wise'|'coordinate_wise'.
            'no_clip': no clipping, standart sgd;
            'norm': standard clipping, \min\{1,\lambda / \|\nabla f(x)\|\} \nabla f(x);
            'layer_wise': standard clipping, but it is calculated for each layer independently;
            'coordinate_wise': coordinate wise clipping,
                \min\{1_n, \lambda / \nabla f(x)\} \nabla f(x)
            where all operations are coordinate wise. (Default: 'norm')
        clipping_level (float, optional):
        level of clipping \lambda (see clipping_type for more information).
            In this variant the clipping level changes, and clipping_level is used to calculate it
            Defaults value depends on clipping_type:
            for clipping_type='norm' default clipping_level=1.0
            for clipping_type='layer_wise' default clipping_level=0.3
            for clipping_type='coordinate_wise' default clipping_level=0.1
        nu (int, optional): smoothness, must be in [0,1]
        a_k_ratio_upper_bound (float, optional): maximum A_k / A_{k+1} ratio, must be in [0,1]
            At big steps, A_k / A_{k+1} ratio tends to 1, thus the method becomes too conservative.
            If a_k_ratio_upper_bound < A_k / A_{k+1} we manually set
            A_k = a_k_ratio_upper_bound * A_{k+1}, see code for details
        clipping_iter_start (int, optional): must be > 0.
            If specified, \nu > 0 and clipping_type=='norm' then
        clipping_level will be chosen to ensure that clipping starts at this iteration of method
        (we can find clipping_level B from B / (k^{2\nu/(1+\nu)}\alpha_0) = 1 when \nu > 0)
    Example:
        >>> optimizer = ClippedSSTM(model.parameters(), l_r=0.01, L=10,
                                     clipping_type='norm', clipping_level=10)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(
            self, params,
            l_r=required,
            L=required,
            clipping_type='no_clip',
            clipping_level=_DependingParameter('clipping_type'),
            beta=0,
            nu=1,
            a_k_ratio_upper_bound=1.0,
            clipping_iter_start=None,
            **kwargs
    ):
        if l_r is not required and l_r <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(l_r))
        if L is not required and L < 0.0:
            raise ValueError("Invalid Lipschitz constant: {}".format(l_r))

        type_to_default_level = {
            'no_clip': 0.0,
            'norm': 1.0,
            'layer_wise': 0.3,
            'coordinate_wise': 0.1,
            'auto_clip': 1.0,
            'linear_stoch_norm': 1.0,
            'quadratic_stoch_norm': 1.0,
            'linear_stoch_autoclip': 1.0,
            'quadratic_stoch_autoclip': 1.0
        }

        if clipping_type not in type_to_default_level:
            raise ValueError("Invalid clipping type: {}, possible types are {}".
                             format(clipping_type, list(type_to_default_level.keys())))
        if not isinstance(clipping_level, depending) and clipping_level < 0.0:
            raise ValueError("Invalid clipping level: {}".format(clipping_level))
        if isinstance(clipping_level, depending):
            clipping_level = type_to_default_level[clipping_type]
        if nu < 0.0 or nu > 1.0:
            raise ValueError("Invalid nu: {}".format(nu))
        if a_k_ratio_upper_bound <= 0.0 or a_k_ratio_upper_bound > 1.0:
            raise ValueError("Invalid a_k_ratio_upper_bound: {}".format(a_k_ratio_upper_bound))
        if clipping_iter_start is not None:
            if not isinstance(clipping_iter_start, int) or clipping_iter_start <= 0:
                raise ValueError("Invalid clipping_iter_start: {}, should be positive integer")
            if nu > 0 and clipping_type == 'norm':
                a = 1 / l_r
                # clipping_level / ( 1 / (2 * a * L) * (k + 1) ** (2 * nu / (1 + nu))) = 1
                clipping_level = 1 / (2 * a * L) * (clipping_iter_start + 1) ** (2 * nu / (1 + nu))
                # print(clipping_level)
            elif nu < 1e-4:
                a = 1 / l_r
                clipping_level = clipping_level / (2 * a * L)

        defaults = dict(
            l_r=l_r, L=L,
            clipping_type=clipping_type, clipping_level=clipping_level,
            nu=nu, a_k_ratio_upper_bound=a_k_ratio_upper_bound,
            state=dict()
        )

        kwargs['clipping_type'] = clipping_type
        kwargs['clipping_level'] = clipping_level
        kwargs['beta'] = beta

        self.grad_desc_step = get_clipped_grad_desc_step(**kwargs)
        super(ClippedSSTM, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(ClippedSSTM, self).__setstate__(state)

    @torch.no_grad()  # sets all requires_grad flags to False
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        g_list = []
        x_list = []

        for group in self.param_groups:
            d_p_list = []
            L = group['L']
            a = 1 / group['l_r']
            clipping_type = group['clipping_type']
            clipping_level = group['clipping_level']
            nu = group["nu"]
            a_k_ratio_upper_bound = group["a_k_ratio_upper_bound"]

            state = group['state']
            # lazy state initialization
            if len(state) == 0:
                state['k'] = 0
                state['alpha_k_1'] = 0
                state['lambda_k_1'] = 0
                state['A_k'] = 0
                state['A_k_1'] = 0

                state['y_k'] = []
                state['z_k'] = []
                for p in group['params']:
                    if p.grad is not None:
                        state['y_k'].append(p.detach().clone())
                        state['z_k'].append(p.detach().clone())

            k = state['k']
            alpha_k_1 = state['alpha_k_1']
            lambda_k_1 = state['lambda_k_1']
            A_k = state['A_k']
            A_k_1 = state['A_k_1']
            y_k = [y.detach().clone() for y in state['y_k']]
            z_k = [z.detach().clone() for z in state['z_k']]

            if k > 0:
                for p in group['params']:
                    if p.grad is not None:
                        d_p_list.append(p.grad.data)
                        g_list.append(p.grad.data)
                        x_list.append(p.data)

                # update z_{k+1}
                self.grad_desc_step(
                    z_k,
                    d_p_list,
                    None,  # no momentum history
                    alpha_k_1,
                    0,  # no momentum, thus 0
                    clipping_type,
                    lambda_k_1
                )

                # update y_{k+1}
                i = 0
                for p in group['params']:
                    if p.grad is not None:
                        y_k[i].data = (A_k * y_k[i].data + alpha_k_1 * z_k[i].data) / A_k_1
                        i += 1

            # k_1 means "k + 1", so alpha_k_1 means \alpha_{k+1}
            alpha_k_1 = 1 / (2 * a * L) * (k + 1) ** (2 * nu / (1 + nu))

            A_k = state['A_k_1']
            A_k_1 = A_k + alpha_k_1

            # apply upper bound on A_k / A_{k+1} ratio
            if a_k_ratio_upper_bound < 1.0:
                ratio_mul_factor = 1.0 / (1.0 - a_k_ratio_upper_bound)
                if A_k > ratio_mul_factor * alpha_k_1:
                    A_k = (ratio_mul_factor - 1.0) * alpha_k_1
                    A_k_1 = ratio_mul_factor * alpha_k_1

            lambda_k_1 = clipping_level / alpha_k_1
            # lambda_k_1 = clipping_level

            state['y_k'] = y_k
            state['z_k'] = z_k

            # update x_{k+1}
            i = 0
            for p in group['params']:
                if p.grad is not None:
                    p.data = (A_k * state['y_k'][i].data + alpha_k_1 * state['z_k'][i].data) / A_k_1
                    i += 1

            state['k'] += 1
            state['alpha_k_1'] = alpha_k_1
            state['lambda_k_1'] = lambda_k_1
            state['A_k'] = A_k
            state['A_k_1'] = A_k_1

        return loss, g_list, x_list


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
                for i in range(len(g2)):
                    G += (g2[i] - g1[i]).norm() ** 2
                    X += (x2[i] - x1[i]).norm() ** 2
                G **= 0.5
                X **= 0.5
                L = G / X

                L_history.append(L.detach().cpu())

            optimizer.zero_grad()

            g1 = deepcopy(g2)
            x1 = deepcopy(x2)

    best_L = np.array(L_history).mean() * 2

    return best_L, L_history
