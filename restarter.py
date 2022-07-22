import torch
from copy import deepcopy
from typing import List

from .optimizers_collector import OptimizerProperties
from .utils import recursive_to


class Restarter:
    """Restarter class for saving and reboot model parameters
    Args:
        optimizer_properties: OptimizerProperties - optimizer_properties (used for optimizer collector)
        first_restart_steps_cnt: int - the number of steps after which the first restart is performed
        restart_coeff: int - the coefficient by which the number of steps is multiplied until the next restart
        max_steps_cnt: int - the maximum number of steps between restarts, after which restarts are turned off
    """

    def __init__(self, optimizer_properties: OptimizerProperties,
                 first_restart_steps_cnt=100, restart_coeff=1.5, max_steps_cnt=5000) -> None:
        self.optimizer_class = optimizer_properties.optimizer_class
        self.optimizer_kwargs = optimizer_properties.optimizer_kwargs
        self.num_cur_restart = first_restart_steps_cnt
        self.restart_coeff = restart_coeff
        self.max_restart_cnt = max_steps_cnt

        self.coords_in_restart = []

        if torch.cuda.is_available():
            self.main_device = torch.device('cuda')
        else:
            self.main_device = torch.device('cpu')
        self.cpu_device = torch.device('cpu')

    def add_coords(self, coords: List[torch.Tensor]) -> None:
        """ Save cur coords (model.parameters()) to history"""
        self.coords_in_restart.append([])
        append_coords = deepcopy([i for i in coords])
        for param in append_coords:
            param.requires_grad = False
            self.coords_in_restart[-1].append(param.detach().cpu())

    def get_mean_coords(self) -> list:
        """Calculates all mean coordinates in current restart"""
        mean_coords = []
        layer_count = len(self.coords_in_restart[0])
        for layer_ind in range(layer_count):
            mean_coords.append(
                sum([step_model_coords[layer_ind] for step_model_coords in self.coords_in_restart]) / layer_count)
        return mean_coords

    def empty_coords(self) -> None:
        """Deletes all coordinates in self.coords_in_restart list"""
        self.coords_in_restart = []

    def make_restart(self, model, optimizer):
        """Update model parameters and create new optimizer for to continue training
        Args:
            model: torch.nn.Module - model for updating
            optimizer: torch.optim.Optimizer - cur optimizer of model
            (returns if we don't need to make restart)

        Return:
            torch.optim.Optimizer: new optimizer object in case we made restart and current optimizer if we didn't
        """

        if len(self.coords_in_restart) < self.max_restart_cnt:
            if len(self.coords_in_restart) > self.num_cur_restart:
                print(f'Restart: {self.num_cur_restart}')
                mean_coords = self.get_mean_coords()

                for i, param in enumerate(model.parameters()):
                    param.requires_grad = False
                    recursive_to(param, self.cpu_device)
                    param.add_(-param + mean_coords[i])
                    recursive_to(param, self.main_device)
                    param.requires_grad = True

                self.empty_coords()
                new_optimizer = self.optimizer_class(model.parameters(), **self.optimizer_kwargs)
                self.num_cur_restart = int(self.num_cur_restart * self.restart_coeff)
                print('End Restart')
                return new_optimizer

        return optimizer
