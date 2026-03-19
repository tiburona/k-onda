from k_onda.sources import Collection
from pathlib import PurePath
from collections import defaultdict
import pint
import pint_xarray
import yaml
import json

from .subject import Subject
from k_onda.provenance import AnnotatorMixin
from k_onda.utils import recursive_update



# some sample config dictionaries

top_level_config = {
    'root': 'some_path',
     'units_to_set': {
         'raw_sample': (1/30000, 's', 'rs'),
         'lfp_sample': (1/2000, 's', 'ls')
         }
}

subjects_config = {
    '_base': {
        'sessions': ['learning_day_1', 'learning_day_2', 'recall']
    },
    'IG144': {
        'conditions': {'treatment': 'control'},
        },
    'IG145': {
        'conditions': {'treatment': 'defeat'},
        'sessions': ['learning_day_1', 'learning_day_2', 'recall145']
        }
    }

sessions_config = {
    'base': {
        'nev_path': '{experiment.root}/{subject.id}/{session.label}/{session.label}.mat',
        'data_sources': ['phy', 'lfp'],
        'epochs':  {
            'tone': {
                'from_nev': True,
                'code': 65502,
                'duration': 30,
                'conditions': {'stimulus': 'tone'},
            },
            'pretone': {
                'relative_to': 'tone',
                'shift': -30,
                'duration': 30,
                'conditions': {'stimulus': 'pretone'},
            }},
    },
    'learning': {
        'inherits': 'base',
        'label': 'learning'
        },
    'learning_day_1': {
        'inherits': 'learning',
        'label': 'learning_day_1'
        },
    'learning_day_2': {
        'inherits': 'learning',
        'label': 'learning_day_2'
        },
    'recall': {
        'inherits': 'base',
        'label': 'recall'
        },
    'recall145': {
        'inherits': 'recall',
        'epochs': {'tone': {'code': 65503}},
        'data_sources': ['phy', 'lfp_swapped']
    }
}

data_sources_config = {
    'spike': {'path': '{experiment.root}/{subject.id}/{session.label}/spike', 'registry_key': 'phy'},
    'lfp': {'path': '{experiment.root}/{subject.id}/{session.label}/lfp/{session.label}.ns3', 
            'region_to_row': {'bla': 0, 'pl': 1}, 
            'registry_key': 'lfp'},
    'lfp_swapped': {
        'inherits': 'lfp',
        'region_to_row': {'bla': 1, 'pl': 0},
    }
}


class Experiment(AnnotatorMixin):

    _snapshot_fields = ('subject_ids',)
    
    def __init__(
            self, 
            experiment_id, 
            subjects=None, 
            top_level_config=None, 
            subjects_config=None, 
            sessions_config=None,
            data_sources_config=None,
            ):
        self.id = experiment_id
        if subjects is None:
            self.subjects = set()
        else:
            self.subjects = set(subjects)
        self.top_level_config = top_level_config or {}
        self.subjects_config = subjects_config or {}
        self.sessions_config = sessions_config or {}
        self.data_sources_config = data_sources_config or {}
        self.root = None
        self.ureg = None
        self.path_constructor_id = self.id
        self.subject_conditions = defaultdict(dict)
        self._init_annotations()

    @property
    def all_neurons(self):
        return Collection([n for s in self.subjects for n in s.neurons])

    @property
    def subject_ids(self):
        return frozenset([subject.id for subject in self.subjects])
    
    def configure(self, **configs):
        
        for config_label, config in configs.items():

            if isinstance(config, dict):
                ext = None
            elif isinstance(config, str):
                ext = PurePath(config).suffix
            elif isinstance(config, PurePath):
                ext = config.suffix
            else:
                raise TypeError("Unknown type for experiment config.")
            
            if ext == '.yaml':
                config = yaml.safe_load(config)
            elif ext == '.json':
                config = json.loads(config)
            elif ext is None:
                pass
            else:
                raise ValueError("Unknown extension for config")
            
            existing_config = getattr(self, config_label) 
            setattr(self, config_label, recursive_update(existing_config, config))
        

    def initialize(self):
        self.configure_top_level()
        self.create_subjects()
  
    def configure_top_level(self):
        # sample dictionary
        # top_level_config = {
        #   'root': 'some_path',
        #   'units_to_set': {'raw_sample': (1/30000, 's', 'rs')}
        # }
        self.root = self.top_level_config.get('root')

        units_to_set = self.top_level_config.get('units_to_set')
        if units_to_set is not None:
            ureg = pint.UnitRegistry()
            pint.set_application_registry(ureg)
            pint_xarray.setup_registry(ureg)
            for unit_name, (mag, ref_unit, abbrev) in units_to_set.items():
                ureg.define(f'{unit_name} = {mag} * {ref_unit} = {abbrev}')
            self.ureg = ureg

    def create_subjects(self):
        base_config = self.subjects_config.get('_base', {})
        for subject_id, subject_config in self.subjects_config.items():
            if subject_id != '_base':
                subject_config = base_config | subject_config
                self.create_subject(subject_id, subject_config)

    def create_subject(self, subject_id, subject_config):
        subject_sessions = subject_config.get('sessions')
        subject = Subject(subject_id)
        self.subject_conditions[subject_id] = subject_config.get('conditions', {})
        subject.create_sessions(self, subject_sessions)
        self.subjects.add(subject)
        return subject

    def add_subjects(self, subjects, sessions=None):
        # to implement later
        pass
        
    def add_subject(self, subject_id, sessions=None):
        # to implement later
        pass

  
