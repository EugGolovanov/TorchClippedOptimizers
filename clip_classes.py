from typing import List, Optional

import torch
from torch import Tensor

from utils import GradLenHistory

AUTO_CLIP_TYPES = ['auto_clip', 'linear_stoch_autoclip', 'quadratic_stoch_autoclip']


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
        clip = bool(torch.rand(1) < prob)

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
        clip = bool(torch.rand(1) < prob)

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
        clip = bool(torch.rand(1) < prob)

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
        clip = bool(torch.rand(1) < prob)

        if not clip or grad_norm_p > grad_norm:
            return 1
        return min(1, grad_norm_p / grad_norm)
