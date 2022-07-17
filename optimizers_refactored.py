import heapq as hp
from typing import List, Optional
import random

import numpy as np
import torch
from torch import Tensor
from torch.optim import Optimizer


AUTO_CLIP_TYPES = ['auto_clip', 'linear_stoch_autoclip', 'quadratic_stoch_autoclip']


class GradLenHistory:
    def __init__(self, clipping_type: str, p: float = 0.25):
        if p <= 0 or p > 1:  # 0 < p ≤ 1
            raise ValueError('Invalid p value (expected value between 0 and 1)')
        self.p = p
        self.clipping_type = clipping_type
        self.heap = []
        hp.heapify(self.heap)
        self.all_grad_lens = []

    def get_grad_len(self):
        return -self.heap[0]

    def add_grad_len(self, grad):
        self.all_grad_lens.append(grad)
        if self.clipping_type in AUTO_CLIP_TYPES:
            if len(self.heap) / len(self.all_grad_lens) < self.p:
                hp.heappush(self.heap, -grad)
            elif grad < -self.heap[0]:
                hp.heappushpop(self.heap, -grad)

    def get_history(self) -> list:
        return self.all_grad_lens


class NoClip:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        if kwargs['clipping_type'] in AUTO_CLIP_TYPES:
            self.grad_history = GradLenHistory(kwargs['clipping_type'], kwargs['p_autoclip'])
        else:
            self.grad_history = GradLenHistory(kwargs['clipping_type'])

    def get_alpha(self, **kwargs):
        return 1

    def __call__(self,
                 params: List[Tensor],
                 d_p_list: List[Tensor],
                 momentum_buffer_list: List[Optional[Tensor]],
                 lr: float,
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

            alpha = self.get_alpha(d_p=d_p, grad_norm=grad_norm, clipping_level=clipping_level, beta=self.kwargs['beta'])
            if clipping_type == 'coordinate_wise':
                param.add_(-lr * alpha * d_p)
            else:
                param.add_(d_p, alpha=-lr * alpha)


class NormClip(NoClip):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_alpha(self, **kwargs):
        clipping_level = kwargs['clipping_level']
        grad_norm = kwargs['grad_norm']
        return min(1, clipping_level / grad_norm)


class LinearStochNormClip(NoClip):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_alpha(self, **kwargs):
        beta = kwargs['beta']
        clipping_level = kwargs['clipping_level']
        grad_norm = kwargs['grad_norm']
        prob = pow(beta, clipping_level / grad_norm)
        clip = random.choices([True, False], weights=[prob, 1 - prob])[0]
        if not clip or clipping_level > grad_norm:
            return 1
        else:
            return min(1, clipping_level / grad_norm)


class QuadraticStochNormClip(NoClip):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_alpha(self, **kwargs):
        beta = kwargs['beta']
        clipping_level = kwargs['clipping_level']
        grad_norm = kwargs['grad_norm']
        prob = pow(beta, (clipping_level / grad_norm) ** 2)
        clip = random.choices([True, False], weights=[prob, 1 - prob])[0]
        if not clip or clipping_level > grad_norm:
            return 1
        else:
            return min(1, clipping_level / grad_norm)


class LayerWiseClip(NoClip):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def get_alpha(**kwargs):
        clipping_level = kwargs['clipping_level']
        d_p = kwargs['d_p']
        return min(1, clipping_level / d_p.norm())


class CoordWiseClip(NoClip):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def get_alpha(**kwargs):
        eps = 1e-8
        clipping_level = kwargs['clipping_level']
        d_p = kwargs['d_p']
        return torch.clip(clipping_level / (torch.abs(d_p) + eps), min=0, max=1)


class AutoClip(NoClip):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_alpha(self, **kwargs):
        grad_norm_p = self.grad_history.get_grad_len()
        grad_norm = kwargs['grad_norm']
        return min(1, grad_norm_p / grad_norm)


class LinearStochAutoClip(NoClip):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_alpha(self, **kwargs):
        beta = kwargs['beta']
        grad_norm_p = self.grad_history.get_grad_len()
        grad_norm = kwargs['grad_norm']
        prob = pow(beta, grad_norm_p / grad_norm)
        clip = random.choices([True, False], weights=[prob, 1 - prob])[0]
        if not clip or grad_norm_p > grad_norm:
            return 1
        else:
            return min(1, grad_norm_p / grad_norm)


class QuadraticStochAutoClip(NoClip):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_alpha(self, **kwargs):
        beta = kwargs['beta']
        grad_norm_p = self.grad_history.get_grad_len()
        grad_norm = kwargs['grad_norm']
        prob = pow(beta, (grad_norm_p / grad_norm) ** 2)
        clip = random.choices([True, False], weights=[prob, 1 - prob])[0]
        if not clip or grad_norm_p > grad_norm:
            return 1
        else:
            return min(1, grad_norm_p / grad_norm)


def get_clipped_grad_desc_step(**kwargs):
    if kwargs['clipping_type'] == 'no_clip':
        return NoClip(**kwargs)
    elif kwargs['clipping_type'] == 'norm':
        return NormClip(**kwargs)
    elif kwargs['clipping_type'] == 'layer_wise':
        return LayerWiseClip(**kwargs)
    elif kwargs['clipping_type'] == 'coordinate_wise':
        return CoordWiseClip(**kwargs)
    elif kwargs['clipping_type'] == 'auto_clip':
        return AutoClip(**kwargs)
    elif kwargs['clipping_type'] == 'linear_stoch_autoclip':
        return LinearStochAutoClip(**kwargs)
    elif kwargs['clipping_type'] == 'quadratic_stoch_autoclip':
        return QuadraticStochAutoClip(**kwargs)
    elif kwargs['clipping_type'] == 'linear_stoch_norm':
        return LinearStochNormClip(**kwargs)
    elif kwargs['clipping_type'] == 'quadratic_stoch_norm':
        return QuadraticStochNormClip(**kwargs)
    else:
        raise TypeError(f'No clipping type called {kwargs["clipping_type"]}')


class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""

    def __repr__(self):
        return "<required parameter>"


required = _RequiredParameter()


class _DependingParameter(object):
    """Singleton class representing a parameter that depends on other for an Optimizer."""

    def __init__(self, other_parameter_name):
        self.other_parameter_name = other_parameter_name

    def __repr__(self):
        return "<depends on {}>".format(self.other_parameter_name)


depending = _DependingParameter


class clipped_SGD(Optimizer):
    r"""Implements clipped version of stochastic gradient descent
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        clipping_type (string, optional): type of clipping to use: 'norm'|'layer_wise'|'coordinate_wise'.
            'no_clip': no clipping, standart sgd;
            'norm': standard clipping, \min\{1,\lambda / \|\nabla f(x)\|\} \nabla f(x);
            'layer_wise': standard clipping but it is calculated for each layer independently;
            'coordinate_wise': coordinate wise clipping, \min\{1_n, \lambda / \nabla f(x)\} \nabla f(x)
            where all operations are coordinate wise. (Default: 'norm')
        clipping_level (float, optional): level of clipping \lambda (see clipping_type for more information).
            Default value depends on clipping_type:
            for clipping_type='norm' default clipping_level=1
            for clipping_type='layer_wise' default clipping_level=1
            for clipping_type='coordinate_wise' default clipping_level=0.1
    Example:
        >>> optimizer = torch.optim.clipped_SGD(model.parameters(), lr=0.01,
                                                clipping_type='layer_wise', clipping_level=10)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(
            self, params,
            lr=required,
            momentum=0,
            clipping_type='no_clip',
            clipping_level=depending('clipping_type'),
            beta=0,
            **kwargs
    ):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
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
                             format(lr, list(type_to_default_level.keys())))
        if not isinstance(clipping_level, depending) and clipping_level < 0.0:
            raise ValueError("Invalid clipping level: {}".format(clipping_level))
        if isinstance(clipping_level, depending):
            clipping_level = type_to_default_level[clipping_type]
        defaults = dict(
            lr=lr,
            momentum=momentum,
            clipping_type=clipping_type, clipping_level=clipping_level
        )

        kwargs['clipping_type'] = clipping_type
        kwargs['clipping_level'] = clipping_level
        kwargs['beta'] = beta

        self.grad_desc_step = get_clipped_grad_desc_step(**kwargs)
        super(clipped_SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(clipped_SGD, self).__setstate__(state)

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
            lr = group['lr']
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
                lr,
                momentum,
                clipping_type,
                clipping_level
            )

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss

