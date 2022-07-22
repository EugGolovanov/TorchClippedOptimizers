from typing import List

import torch

"""
Example:

from optimizers_collector import OptimizerProperties, OptimizersCollector
optimizers_properties = [
  OptimizerProperties(optimizer_class=clipped_SGD, lr=5e-2, momentum=0.9, 
                    clipping_type="norm", clipping_level=1, p_autoclip=0.75),
  OptimizerProperties(optimizer_class=clipped_SGD, lr=5e-2, momentum=0.9, 
                clipping_type="auto_clip", clipping_level=1, p_autoclip=0.25),
  OptimizerProperties(optimizer_class=optim.SGD, lr=0.01, momentum=0.9),
  OptimizerProperties(optimizer_class=optim.Adam, lr=1e-3)
]

collector = OptimizersCollector(models.resnet18, optimizers_properties)

criterion = nn.CrossEntropyLoss()
opts = collector.get_optimizer_objects()
opt_names = collector.get_names_optimizers()
nets = collector.nets
bs_muls = collector.bs_muls
lr_decays = collector.lr_decays
"""


class OptimizerProperties:
    """ A class for storing the class and initialization parameters of the optimizer
        Args:
            optimizer_class (class/func): callable object, what return optimizer object
            kwargs: all kwargs passed to the initialization call of the optimizer
    """

    def __init__(self, optimizer_class, **kwargs):
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = kwargs


class ModelProperties:
    """ A class for storing the class and initialization parameters of the model
        Args:
            optimizer_class (class/func): callable object, what return model object
            kwargs: all kwargs passed to the initialization call of the model
    """

    def __init__(self, model_class, **kwargs):
        self.model = model_class
        self.model_kwargs = kwargs

    def get_model(self):
        """ Init a new model object and returns it
            Kwargs: all kwargs go to __init__ method
            Return: model object
        """
        return self.model(**self.model_kwargs)


class RestartProperties:
    """A class for storing the class and initialization parameters of the restart object"""

    def __init__(self, restart_class, first_restart_steps_cnt=100, restart_coeff=1.5, max_steps_cnt=5000):
        self.restart_class = restart_class
        self.first_restart_steps_cnt = first_restart_steps_cnt
        self.restart_coeff = restart_coeff
        self.max_steps_cnt = max_steps_cnt

    def get_kwargs(self):
        return {"first_restart_steps_cnt": self.first_restart_steps_cnt,
                "restart_coeff": self.restart_coeff,
                "max_steps_cnt": self.max_steps_cnt}


class OptimizersCollector:
    """
        A class for initializing optimizers and getting their objects
        Args:
            model_properties (ModelProperties object): object for init model
            optimizers_properties (List[OptimizerProperties]): list of optimizer properties for init optimizers
            starting_point_random_seed, history_random_seed (int): seeds for random torch
        Kwargs:
            bs_muls and lr_decays can be passed to kwargs,
            by default they are automatically initialized with ones
    """

    def __init__(self, model_properties: ModelProperties, optimizers_properties: List[OptimizerProperties],
                 start_point_random_seed=42, history_random_seed=42, **kwargs):
        self.optimizers_properties = optimizers_properties
        self.nets = []

        self.bs_muls = [1] * len(self.optimizers_properties)
        self.lr_decays = [1] * len(self.optimizers_properties)

        if "bs_mul" in kwargs.keys():
            self.bs_mul = kwargs["bs_mul"]

        if "lr_decays" in kwargs.keys():
            self.lr_decays = kwargs["lr_decays"]

        for _ in range(len(self.optimizers_properties)):
            torch.manual_seed(start_point_random_seed)
            self.nets.append(model_properties.get_model())
            self.nets[-1].zero_grad()
            self.nets[-1].train()
        torch.manual_seed(history_random_seed)

        self.opts = None
        self.init_optimizers_objects()

        self.opt_names = None
        self.init_optimizer_names()

    def init_optimizer_names(self):
        """Init optimizers names in self.opt_names"""
        self.opt_names = []

        for i in range(len(self.optimizers_properties)):
            name = f"{i+1}) {self.optimizers_properties[i].optimizer_class.__name__};  "
            for k, v in self.optimizers_properties[i].optimizer_kwargs.items():
                name += f"{k}={v}, "
            name = name[:-2]

            self.opt_names.append(name)

    def get_names_optimizers(self):
        """Returns names of optimizers List[str]"""
        return self.opt_names

    def init_optimizers_objects(self):
        """Init optimizers objects in self.opts"""
        self.opts = [i.optimizer_class(self.nets[ind].parameters(), **i.optimizer_kwargs)
                     for ind, i in enumerate(self.optimizers_properties)]

    def get_optimizer_objects(self):
        """Returns optimizers objects List[torch.optim.Optimizer]"""
        return self.opts

    def get_nets(self):
        """Returns models objects List[torch.nn.Module]"""
        return self.nets


class OptimizersCollectorWithRestarts(OptimizersCollector):
    """A class for initializing optimizers and restarters for training
    Args:
        model_properties: ModelProperties - model properties for init
        optimizers_properties: List[OptimizerProperties] - optimizer properties for each optimizer
        restart_properties: List[RestartProperties] - restart properties for each restarters
        start_point_random_seed, history_random_seed: int - random seed for Collector init
    """

    def __init__(self, model_properties: ModelProperties, optimizers_properties: List[OptimizerProperties],
                 restart_properties: List[RestartProperties], start_point_random_seed=42, history_random_seed=42,
                 **kwargs):
        super().__init__(model_properties, optimizers_properties,
                         start_point_random_seed, history_random_seed, **kwargs)

        self.restart_properties = restart_properties
        self.restarters = [restart_property.restart_class(opt_prop, **restart_property.get_kwargs())
                           if restart_property else None
                           for restart_property, opt_prop
                           in zip(self.restart_properties, optimizers_properties)]

    def get_restarters(self):
        """Returns restarters objects"""
        return self.restarters
