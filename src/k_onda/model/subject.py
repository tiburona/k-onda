from collections import defaultdict


class Subject:
    def __init__(self, subject_id):
        self.subject_id = subject_id
        self.sessions = []
        self.data_identities = defaultdict(list)

    def add_to_experiment(self, experiment):
        experiment.subjects.append(self)

    @property
    def experiments(self):
        return list({session.experiment for session in self.sessions})

    @property
    def neurons(self):
        return self.data_identities["neuron"]
