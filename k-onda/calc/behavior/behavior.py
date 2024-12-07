from data.data import Data
from data.period_event import Period, Event

# TODO: give behavior data the hierarchy it needs


class BehaviorPrepMethods:

    @property
    def imported_behavior(self):
        return self._imported_behavior
    
    
    def behavior_prep(self):
        self.prep_data()
        self.prepare_periods()

    def prep_data(self):
        data_importer = self.load_importer()
        self._processed_behavior = data_importer.import_data()

    def load_importer(self):
        
        user_module = self.load_user_module(self.calc_opts.get('behavior_data_importer'))
        importer_class = getattr(user_module, 'MyBehaviorImporter', None)
        if importer_class is None:
            raise ValueError("User plugin must define MyBehaviorImporter")
        importer = importer_class()
        return importer



class BehaviorMethods:

    def get_behavior(self):
        pass   


class BehaviorPeriod(Period):

    def get_behavior(self):
        period_data = self.imported_data[self.period_type][self.identifier]
        if isinstance(period_data, 'dict') and 'events' in period_data:
            return self.get_average('get_behavior')
        else:
            return period_data


class BehaviorEvent(Event):
    
    def get_behavior(self):
        event_data = self.imported_data[self.period_type][self.period.identifier][self.identifier]
        return event_data
    




