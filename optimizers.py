"""
TODO: Module docstring
"""

from torch.optim import Optimizer

from clip_classes import *
from utils import _RequiredParameter, _DependingParameter, get_clipped_grad_desc_step

required = _RequiredParameter()

depending = _DependingParameter('clipping_type')


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
        >>>                                    clipping_type='layer_wise', clipping_level=10)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(
            self, params,
            l_r=required,
            momentum=0,
            clipping_type='no_clip',
            clipping_level=depending,
            beta=0,
            **kwargs
    ):
        if l_r is not required and l_r < 0.0:
            raise ValueError(f"Invalid learning rate: {l_r}")
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
            raise ValueError(f"Invalid clipping type: {l_r}, "
                             f"possible types are {list(type_to_default_level.keys())}")
        if not isinstance(clipping_level, _DependingParameter) and clipping_level < 0.0:
            raise ValueError(f"Invalid clipping level: {clipping_level}")
        if isinstance(clipping_level, _DependingParameter):
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
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

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
            clipping_level=depending,
            beta=0,
            nu=1,
            a_k_ratio_upper_bound=1.0,
            clipping_iter_start=None,
            **kwargs
    ):
        if l_r is not required and l_r <= 0.0:
            raise ValueError(f"Invalid learning rate: {l_r}")
        if L is not required and L < 0.0:
            raise ValueError(f"Invalid Lipschitz constant: {L}")

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
            raise ValueError(f"Invalid clipping type: {clipping_type},"
                             f"possible types are {list(type_to_default_level.keys())}")
        if not isinstance(clipping_level, _DependingParameter) and clipping_level < 0.0:
            raise ValueError(f"Invalid clipping level: {clipping_level}")
        if isinstance(clipping_level, _DependingParameter):
            clipping_level = type_to_default_level[clipping_type]
        if nu < 0.0 or nu > 1.0:
            raise ValueError(f"Invalid nu: {nu}")
        if a_k_ratio_upper_bound <= 0.0 or a_k_ratio_upper_bound > 1.0:
            raise ValueError(f"Invalid a_k_ratio_upper_bound: {a_k_ratio_upper_bound}")
        if clipping_iter_start is not None:
            if not isinstance(clipping_iter_start, int) or clipping_iter_start <= 0:
                raise ValueError(f"Invalid clipping_iter_start: "
                                 f"{clipping_iter_start}, should be positive integer")
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
            state={}
        )

        kwargs['clipping_type'] = clipping_type
        kwargs['clipping_level'] = clipping_level
        kwargs['beta'] = beta

        self.grad_desc_step = get_clipped_grad_desc_step(**kwargs)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

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
