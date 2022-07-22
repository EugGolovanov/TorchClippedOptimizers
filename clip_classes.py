"""Submodule containing clipping classes with clipping and gradient coefficients calculation
"""

import torch
from torch import Tensor
from typing import List, Optional

from .utils import GradHistory, AUTO_CLIP_TYPES


class NoClip:
    """Class-wrapper for calculating gradient coefficient with no-clip version by default
    Keyword Arguments:
        clipping_type (string): type of clipping defining appropriate get_alpha() method
        clipping_level (float): coefficient for constant clipping methods
        p_autoclip (float): p-th percentile for auto-clip clipping methods
        beta (float): basis of clipping probability in random clipping methods
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        if kwargs['clipping_type'] in AUTO_CLIP_TYPES:
            self.grad_history = GradHistory(kwargs['clipping_type'], kwargs['p_autoclip'])
        else:
            self.grad_history = GradHistory(kwargs['clipping_type'])

    def get_alpha(self, **kwargs):
        """Method that calculates clipping coefficient with no-clip version
        Keyword Arguments:
            clipping_level (float): coefficient for constant clipping methods
            p_autoclip (float): p-th percentile for auto-clip clipping methods
            beta (float): basis of clipping probability in random clipping methods
        """
        return 1

    def __call__(self,
                 params: List[Tensor],
                 d_p_list: List[Tensor],
                 momentum_buffer_list: List[Optional[Tensor]],
                 lr: float,
                 momentum: float,
                 clipping_type: str,
                 clipping_level: float):
        """Functional API that performs clipped step for slipped-SGD and clipped-SSTM algorithm
        computation.
        See :class:`clipped_SGD` or class:`clipped_SSTM` for details.
        """

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
                param.add_(-lr * alpha * d_p)
            else:
                param.add_(d_p, alpha=-lr * alpha)


class NormClip(NoClip):
    """Inheritor class for calculating clipping coefficient with norm-clip version
    Keyword Arguments:
        clipping_level (float): coefficient for constant clipping methods
        p_autoclip (float): p-th percentile for auto-clip clipping methods
        beta (float): basis of clipping probability in random clipping methods
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_alpha(self, **kwargs):
        """Method that calculates clipping coefficient with norm-clip version
        """
        clipping_level = kwargs['clipping_level']
        grad_norm = kwargs['grad_norm']
        return min(1, clipping_level / grad_norm)


class LinearRandNormClip(NoClip):
    """Inheritor class for calculating clipping coefficient with linear-random-norm version
    Keyword Arguments:
        clipping_level (float): coefficient for constant clipping methods
        p_autoclip (float): p-th percentile for auto-clip clipping methods
        beta (float): basis of clipping probability in random clipping methods
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_alpha(self, **kwargs):
        """Method that calculates clipping coefficient with linear-random-norm version
        """
        beta = kwargs['beta']
        clipping_level = kwargs['clipping_level']
        grad_norm = kwargs['grad_norm']

        prob = pow(beta, clipping_level / grad_norm)
        clip = bool(torch.rand(1) < prob.cpu())

        if not clip or clipping_level > grad_norm:
            return 1
        return min(1, clipping_level / grad_norm)


class QuadraticRandNormClip(NoClip):
    """Inheritor class for calculating clipping coefficient with quadratic-random-norm version
    Keyword Arguments:
        clipping_level (float): coefficient for constant clipping methods
        p_autoclip (float): p-th percentile for auto-clip clipping methods
        beta (float): basis of clipping probability in random clipping methods
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_alpha(self, **kwargs):
        """Method that calculates clipping coefficient with quadratic-random-norm version
        """
        beta = kwargs['beta']
        clipping_level = kwargs['clipping_level']
        grad_norm = kwargs['grad_norm']

        prob = pow(beta, (clipping_level / grad_norm) ** 2)
        clip = bool(torch.rand(1) < prob.cpu())

        if not clip or clipping_level > grad_norm:
            return 1
        return min(1, clipping_level / grad_norm)


class LayerWiseClip(NoClip):
    """Inheritor class for calculating clipping coefficient with layer-wise version
    Keyword Arguments:
        clipping_level (float): coefficient for constant clipping methods
        p_autoclip (float): p-th percentile for auto-clip clipping methods
        beta (float): basis of clipping probability in random clipping methods
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def get_alpha(**kwargs):
        """Method that calculates clipping coefficient with layer-wise version
        """
        clipping_level = kwargs['clipping_level']
        d_p = kwargs['d_p']
        return min(1, clipping_level / d_p.norm())


class CoordWiseClip(NoClip):
    """Inheritor class for calculating clipping coefficient with coordinate-wise version
    Keyword Arguments:
        clipping_level (float): coefficient for constant clipping methods
        p_autoclip (float): p-th percentile for auto-clip clipping methods
        beta (float): basis of clipping probability in random clipping methods
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def get_alpha(**kwargs):
        """Method that calculates clipping coefficient with coordinate-wise version
        """
        eps = 1e-8
        clipping_level = kwargs['clipping_level']
        d_p = kwargs['d_p']
        return torch.clip(clipping_level / (torch.abs(d_p) + eps), min=0, max=1)


class AutoClip(NoClip):
    """Inheritor class for calculating clipping coefficient with auto-clip version
    Keyword Arguments:
        clipping_level (float): coefficient for constant clipping methods
        p_autoclip (float): p-th percentile for auto-clip clipping methods
        beta (float): basis of clipping probability in random clipping methods
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_alpha(self, **kwargs):
        """Method that calculates clipping coefficient with auto-clip version
        """
        print('Here!')
        grad_norm_p = self.grad_history.get_grad_len()
        grad_norm = kwargs['grad_norm']
        return min(1, grad_norm_p / grad_norm)


class LinearRandAutoClip(NoClip):
    """Inheritor class for calculating clipping coefficient with linear-random-auto-clip version
    Keyword Arguments:
        clipping_level (float): coefficient for constant clipping methods
        p_autoclip (float): p-th percentile for auto-clip clipping methods
        beta (float): basis of clipping probability in random clipping methods
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_alpha(self, **kwargs):
        """Method that calculates clipping coefficient with linear-random-auto-clip version
        """
        beta = kwargs['beta']
        grad_norm_p = self.grad_history.get_grad_len()
        grad_norm = kwargs['grad_norm']

        prob = pow(beta, grad_norm_p / grad_norm)
        clip = bool(torch.rand(1) < prob.cpu())

        if not clip or grad_norm_p > grad_norm:
            return 1
        return min(1, grad_norm_p / grad_norm)


class QuadraticRandAutoClip(NoClip):
    """Inheritor class for calculating clipping coefficient with quadratic-random-auto-clip version
    Keyword Arguments:
        clipping_level (float): coefficient for constant clipping methods
        p_autoclip (float): p-th percentile for auto-clip clipping methods
        beta (float): basis of clipping probability in random clipping methods
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_alpha(self, **kwargs):
        """Method that calculates clipping coefficient with quadratic-random-auto-clip version
        """
        beta = kwargs['beta']
        grad_norm_p = self.grad_history.get_grad_len()
        grad_norm = kwargs['grad_norm']

        prob = pow(beta, (grad_norm_p / grad_norm) ** 2)
        clip = bool(torch.rand(1) < prob.cpu())

        if not clip or grad_norm_p > grad_norm:
            return 1
        return min(1, grad_norm_p / grad_norm)


def get_clipped_grad_desc_step(**kwargs):
    """Returns appropriate clipping class according to provided clipping method
    Keyword arguments:
        clipping_type (string): type of clipping defining appropriate get_alpha() method
        clipping_level (float): coefficient for constant clipping methods
        p_autoclip (float): p-th percentile for auto-clip clipping methods
        beta (float): basis of clipping probability in random clipping methods
    """
    type_class = {
        'no_clip': NoClip,
        'norm': NormClip,
        'layer_wise': LayerWiseClip,
        'coordinate_wise': CoordWiseClip,
        'auto_clip': AutoClip,
        'linear_rand_autoclip': LinearRandAutoClip,
        'quadratic_rand_autoclip': QuadraticRandAutoClip,
        'linear_rand_norm': LinearRandNormClip,
        'quadratic_rand_norm': QuadraticRandNormClip
    }

    try:
        clipping_class = type_class[kwargs['clipping_type']]
    except KeyError:
        raise TypeError(f'No clipping type called {kwargs["clipping_type"]}')
    else:
        return clipping_class(**kwargs)
