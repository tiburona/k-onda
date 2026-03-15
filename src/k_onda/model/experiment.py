from k_onda.sources import Collection


class Experiment:
    
    def __init__(self, experiment_id, subjects=None):
        self.experiment_id = experiment_id
        if subjects is None:
            self.subjects = []
        else:
            self.subjects = subjects

    @property
    def all_neurons(self):
        return Collection([n for s in self.subjects for n in s.neurons])