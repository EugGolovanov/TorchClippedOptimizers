"""Submodule containing clipping classes with clipping and gradient coefficients calculation
"""


from typing import List, Optional, Tuple
from copy import deepcopy

import torch
from torch import Tensor

from utils import GradHistory, AUTO_CLIP_TYPES
from optimizers_collector import OptimizerProperties


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


class Restarts:
    def __init__(self, optimizer_properties: OptimizerProperties,
                 num_cur_restart=100, restart_coeff=1.5, max_restart_cnt=5000) -> None:
        self.optimizer_class = optimizer_properties.optimizer_class
        self.optimizer_kwargs = optimizer_properties.optimizer_kwargs
        self.num_cur_restart = num_cur_restart
        self.restart_coeff = restart_coeff
        self.max_restart_cnt = max_restart_cnt

        self.coords_in_restart = []

        if torch.cuda.is_available():
            self.main_device = torch.device('cuda')
        else:
            self.main_device = torch.device('cpu')
        self.cpu_device = torch.device('cpu')

    def add_coords(self, coords: List[torch.Tensor]) -> None:
        self.coords_in_restart.append([])
        append_coords = deepcopy([i for i in coords])
        for param in append_coords:
            param.requires_grad = False
            self.coords_in_restart[-1].append(param.detach().cpu())

    def get_mean_coords(self) -> list:
        """Calculates all mean coordinates in current restart
        """
        mean_coords = []
        layer_count = len(self.coords_in_restart[0])
        for layer_ind in range(layer_count):
            mean_coords.append(sum([step_model_coords[layer_ind] for step_model_coords in self.coords_in_restart]) / layer_count)
        return mean_coords

    def empty_coords(self) -> None:
        """Deletes all coordinates in self.coords_in_restart list
        """
        self.coords_in_restart = []

    def make_restart(self, model, optimizer):
        if len(self.coords_in_restart) < self.max_restart_cnt:
            if len(self.coords_in_restart) >= self.num_cur_restart:

                print(f'Restart: {self.num_cur_restart}')
                mean_coords = self.get_mean_coords()

                for i, param in enumerate(model.parameters()):
                    param.requires_grad = False
                    param.to(self.cpu_device)
                    mean_coords[i].to(self.cpu_device)
                    param.add_(-param + mean_coords[i])
                    param.requires_grad = True
                    param.to(self.main_device)
                self.empty_coords()
                new_optimizer = self.optimizer_class(model.parameters(), **self.optimizer_kwargs)
                self.num_cur_restart = int(self.num_cur_restart * self.restart_coeff)
                print('End restart')
                return new_optimizer

        return optimizer
