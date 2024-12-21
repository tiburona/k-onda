from data import Data

# TODO: give behavior data the hierarchy it needs

class Behavior:
    def __init__(self, experiment, info, data):
        self.experiment = experiment
        self.all_animals = []
        self.initialize(data)

    def initialize(self, data):

        for animal in self.experiment.all_animals:
            if animal.identifier in data:
                self.all_animals.append(BehaviorAnimal(animal, data[animal.identifier]))

    @property
    def all_periods(self):
        return [period for animal in self.all_animals for period in animal.all_periods]


class BehaviorAnimal(Data):
    name = 'animal'

    def __init__(self, animal, data):
        self.spike_target = animal
        self.periods = {period_type: [BehaviorPeriod(self, period_type, i, behavior_data)
                                      for i, behavior_data in enumerate(data[period_type])]
                        for period_type in self.spike_target.period_info}
        self.all_periods = [period for period_type in self.periods for period in self.periods[period_type]]

    def __getattr__(self, name):
        return getattr(self.spike_target, name)


class BehaviorPeriod(Data):
    name = 'period'

    def __init__(self, animal, period_type, i, data):
        self.val = data
        self.period_type = period_type
        self.parent = animal
        self.identifier = i

    def get_percent_freezing(self):
        return self.val

