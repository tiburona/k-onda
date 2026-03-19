from collections import defaultdict

from .session import Session
from k_onda.provenance import AnnotatorMixin
from k_onda.mixins import ConfigSetter


class Subject(AnnotatorMixin, ConfigSetter):

    _snapshot_fields = ('session_ids',)

    def __init__(self, subject_id):
        self.id = subject_id
        self.sessions = []
        self.data_identities = defaultdict(list)
        self.path_constructor_id = self.id
        self._init_annotations()

    @property
    def experiments(self):
        return list({session.experiment for session in self.sessions})
    
    @property
    def session_ids(self):
        return frozenset([session.display_id for session in self.sessions])

    @property
    def neurons(self):
        return self.data_identities["neuron"]

    def create_sessions(self, experiment, subject_sessions):
        sessions = [self.create_session(experiment, session) for session in subject_sessions]
        self.sessions.extend(sessions)
        return sessions

    def create_session(self, experiment, session):
        session_config = self.resolve_config(session, experiment.sessions_config)
        session = Session(experiment, self, session_config)
        self.set_annotation('create_session', session.uid, self)
        return session


    def remove_session(self, session_id):
        self.sessions = [session for session in self.sessions if session.uid != session_id]
        self.set_annotation('remove_session', session_id, self)

    
