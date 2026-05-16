from pathlib import PurePath
from collections import defaultdict
import pint
import pint_xarray
import yaml
import json

from k_onda.sources import Collection
from k_onda.mixins import ConfigSetter
from .subject import Subject
from k_onda.provenance import AnnotatorMixin
from k_onda.utils import recursive_update


class Experiment(AnnotatorMixin, ConfigSetter):
    _snapshot_fields = ("subject_ids",)

    def __init__(
        self,
        experiment_id,
        subjects=None,
        global_config=None,
        top_level_config=None,
        subjects_config=None,
        sessions_config=None,
        data_sources_config=None,
        data_identity_config=None,
        epochs_config=None,
        events_config=None,
    ):
        self.id = experiment_id
        if subjects is None:
            self.subjects = set()
        else:
            self.subjects = set(subjects)
        self.global_config = global_config or {}
        self.top_level_config = top_level_config or {}
        self.subjects_config = subjects_config or {}
        self.sessions_config = sessions_config or {}
        self.data_sources_config = data_sources_config or {}
        self.data_identity_config = data_identity_config or {}
        self.epochs_config = epochs_config or {}
        self.events_config = events_config or {}
        self.root = None
        self.ureg = None
        self.path_constructor_id = self.id
        self.subject_conditions = defaultdict(dict)
        self.configure_global()
        self._init_annotations()

    @classmethod
    def from_config(cls, experiment_id, **kwargs):
        exp = Experiment(experiment_id, **kwargs)
        return exp

    @property
    def all_neurons(self):
        return Collection([n for s in self.subjects for n in s.neurons])

    @property
    def all_lfp_brain_regions(self):
        return Collection([br for s in self.subjects for br in s.lfp_brain_regions])

    @property
    def subject_ids(self):
        return frozenset([subject.id for subject in self.subjects])

    @staticmethod
    def get_ext(config):

        if isinstance(config, dict):
            return None
        elif isinstance(config, str):
            return PurePath(config).suffix
        elif isinstance(config, PurePath):
            return config.suffix
        else:
            raise TypeError("Unknown type for experiment config.")

    def load_config(self, config):
        ext = self.get_ext(config)

        if ext == ".yaml":
            with open(config) as f:
                config = yaml.safe_load(f)
        elif ext == ".json":
            with open(config) as f:
                config = json.load(f)
        elif ext is None:
            pass
        else:
            raise ValueError("Unknown extension for config")

        return config

    def configure_global(self):

        global_config = self.load_config(self.global_config)
        self.configure(**global_config)
        self.resolve_configs()

    def resolve_configs(self):
        for name in [
            "top_level_config",
            "subjects_config",
            "sessions_config",
            "data_sources_config",
            "data_identity_config",
            "epochs_config",
            "events_config",
        ]:
            setattr(self, name, self.load_config(getattr(self, name)))

    def configure(self, **configs):

        for config_label, config in configs.items():
            config = self.load_config(config)
            existing_config = getattr(self, config_label)
            setattr(self, config_label, recursive_update(existing_config, config))

    def initialize(self):
        self.configure_top_level()
        self.create_subjects()
        return self

    def configure_top_level(self):
        # sample dictionary
        # top_level_config = {
        #   'root': 'some_path',
        #   'units_to_set': {'raw_sample': (1/30000, 's', 'rs')}
        # }
        self.root = self.top_level_config.get("root")

        units_to_set = self.top_level_config.get("units_to_set")
        if units_to_set is not None:
            ureg = pint.UnitRegistry()
            pint.set_application_registry(ureg)
            pint_xarray.setup_registry(ureg)

            for unit_name, (mag, ref_unit, abbrev) in units_to_set.items():
                ureg.define(f"{unit_name} = {mag} * {ref_unit} = {abbrev}")

            self.ureg = ureg

    def create_subjects(self):
        for subject_key, subject_config in self.subjects_config.items():
            config = self.resolve_config(subject_config, self.subjects_config)
            if "members" in config:
                for member in config["members"]:
                    self.create_subject(member, config)
            elif not subject_key.startswith("_"):
                self.create_subject(subject_key, config)

    def create_subject(self, subject_id, subject_config=None):
        config = {}
        if 'inherits' in subject_config:
            config_key = subject_config['inherits']
            config_source = self.subjects_config
            config = self.resolve_inheritance(config_key, config_source)
        subject_config = config | (subject_config or {})
        subject_sessions = subject_config.get("sessions", [])
        subject = Subject(subject_id)
        self.subject_conditions[subject_id] = subject_config.get("conditions", {})
        subject.create_sessions(self, subject_sessions)
        self.subjects.add(subject)
        return subject

    def add_subjects(self, subjects, sessions=None):
        # to implement later
        pass

    def add_subject(self, subject, sessions=None):
        # to implement later
        pass
