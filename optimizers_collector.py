import torch
from torch.optim import Adam
from typing import List

"""
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
    """Класс для хранения класса и параметров инициализации оптимизатора"""
    def __init__(self, optimizer_class, **kwargs):
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = kwargs

    def get_optimizer(self, **kwargs):
        return self.optimizer_class(**self.optimizer_kwargs, **kwargs)


class ModelProperties:
    def __init__(self, model_class, **kwargs):
        self.model = model_class
        self.model_kwargs = kwargs

tmp = OptimizerProperties(optimizer_class=Adam)

class OptimizersCollector:
    def __init__(self, model_properties: ModelProperties, optimizers_properties: List[OptimizerProperties],
                 starting_point_random_seed=42, history_random_seed=42, **kwargs):
        """
            Класс для инициализации оптимизаторов и получения их параметров
            bs_muls и lr_decays можно передать в kwargs,
            по-умолчанию автоматически инициализируются единичками
        """
        self.optimizers_properties = optimizers_properties
        self.nets = []

        self.bs_muls = [1] * len(self.optimizers_properties)
        self.lr_decays = [1] * len(self.optimizers_properties)

        if "bs_mul" in kwargs.keys():
            self.bs_mul = kwargs["bs_mul"]

        if "lr_decays" in kwargs.keys():
            self.lr_decays = kwargs["lr_decays"]

        for _ in range(len(self.optimizers_properties)):
            torch.manual_seed(starting_point_random_seed)
            self.nets.append(model_properties.model(**model_properties.model_kwargs))
            self.nets[-1].zero_grad()
            self.nets[-1].train()
        torch.manual_seed(history_random_seed)

    def get_optimizer_objects(self):
        opts = [i.optimizer_class(self.nets[ind].parameters(), **i.optimizer_kwargs)
                for ind, i in enumerate(self.optimizers_properties)]
        return opts

    def get_names_optimizers(self):
        opt_names = []

        for i in range(len(self.optimizers_properties)):
            name = f"{self.optimizers_properties[i].optimizer_class.__name__};  "
            for k, v in self.optimizers_properties[i].optimizer_kwargs.items():
                name += f"{k}={v}, "
            name = name[:-2]
            opt_names.append(name)

        return opt_names
