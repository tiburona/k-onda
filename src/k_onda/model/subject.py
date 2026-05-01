from collections import defaultdict

from .session import Session
from k_onda.provenance import AnnotatorMixin
from k_onda.mixins import ConfigSetter


class Subject(AnnotatorMixin, ConfigSetter):
    _snapshot_fields = ("session_ids",)

    def __init__(self, subject_id):
        self.id = subject_id
        self.sessions = []
        self.session_views = {}
        self.data_identities = defaultdict(list)
        self.path_constructor_id = self.id
        self._init_annotations()

    # what do I want to be able to do:
    # subj.exp_id.lfp.bla

    # also experiment.lfp.bla
    # also experiment.learning_day_1.lfp.bla
    # subject.learning_day_1.lfp.bla
    # subject.lfp.bla.learning_day_1

    # HOWEVER THIS IS A SIDE QUEST!
    # just remember you want to use some combination of __getattr__, simple namespaces,
    # and lightweight view objects to make this happen and disengage for now.

    @property
    def experiments(self):
        return list({session.experiment for session in self.sessions})

    @property
    def session_ids(self):
        return frozenset([session.display_id for session in self.sessions])

    @property
    def neurons(self):
        for session in self.sessions:
            session.ensure_neurons()
        return self.data_identities["neuron"]

    @property
    def lfp_brain_regions(self):
        for session in self.sessions:
            session.ensure_lfp_brain_regions()
        return self.data_identities["lfp_brain_region"]

    def create_sessions(self, experiment, subject_sessions_config=None):
        sessions = []
        sessions_config = self.resolve_config(
            subject_sessions_config, experiment.sessions_config
        )
        for key, config in sessions_config.items():
            session = self.create_session(experiment, config)
            sessions.append(session)
        self.sessions.extend(sessions)
        return sessions

    def create_session(self, experiment, session_config):
        session = Session(experiment, self, session_config)
        self.set_annotation("create_session", session.uid, self)
        return session

    def remove_session(self, session_id):
        self.sessions = [
            session for session in self.sessions if session.uid != session_id
        ]
        # self.session_views = {label: view for label, view in self.session_views
        #                       if view.session.uid != session_id}
        self.set_annotation("remove_session", session_id, self)
