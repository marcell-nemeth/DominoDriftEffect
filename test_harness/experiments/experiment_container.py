from abc import ABC


class ExperimentContainer(ABC):

    def __init__(self):
        self.model_container = {}
        self.dataset_container = {}
        self.iter_container = {}

    def add_model_iteration(self, _experiment, _exp_type):
        self.iter_container.update({_exp_type: _experiment})

    def add_model(self, _n_component):
        self.model_container.update({_n_component: self.iter_container})
        self.iter_container = {}

    def add_dataset(self, _n_component, _dataset):
        print(f'Adding {_n_component} dim. PCA dataset')
        self.dataset_container.update({_n_component: _dataset})
