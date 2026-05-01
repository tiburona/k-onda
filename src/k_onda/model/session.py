from collections import defaultdict
import h5py
from datetime import datetime
from dataclasses import dataclass
from copy import deepcopy
import uuid
import re

from k_onda.mixins import ConfigSetter
from k_onda.utils import group_to_dict
from k_onda.loci import Epoch, EpochSet, EventSet


from k_onda.sources import LFPRecording, PhyOutput, Neuron, LFPBrainRegion

data_source_registry = {"lfp": {"class": LFPRecording}, "phy": {"class": PhyOutput}}


@dataclass(frozen=True)
class TimeBase:
    fs_hz: float
    start_sample: int = 0
    start_datetime: datetime | None = None
    duration_sample: int | None = None
    duration_sec: float | None = None


@dataclass
class Onset:
    in_samples: int | None = None
    in_secs: float | None = None
    duration_sec: float | None = None
    code: int | str | None = None
    label: str | None = None


class NEVMixin:
    # todo: for now we're loading NEV file from converted matfile; need to also be
    # able to load directly of course

    def load_nev(self, nev_path, mat=True):

        nev_path = self.fill_fields(
            nev_path, experiment=self.experiment, subject=self.subject, session=self
        )

        if mat:
            with h5py.File(nev_path, "r") as mat_file:
                data = group_to_dict(mat_file["NEV"])
                return data

    def get_nev_start_time(self, data):
        raw_time = data["MetaTags"]["DateTimeRaw"]
        raw_time = [int(entry) for sublist in raw_time for entry in sublist]
        year, month, _dow, day, hour, minute, second, millisecond = raw_time
        dt = datetime(
            year, month, day, hour, minute, second, microsecond=millisecond * 1000
        )
        return dt

    def get_nev_time_base(self, data):
        fs_hz = data["MetaTags"]["TimeRes"]
        start_datetime = self.get_nev_start_time(data)
        duration_sample = data["MetaTags"]["DataDuration"]
        duration_sec = data["MetaTags"]["DataDurationSec"]
        return TimeBase(
            fs_hz=fs_hz,
            start_datetime=start_datetime,
            duration_sample=duration_sample,
            duration_sec=duration_sec,
        )

    def get_nev_onsets(self, data):

        markers = defaultdict(list)

        for i, code in enumerate(data["Data"]["SerialDigitalIO"]["UnparsedData"][0]):
            in_samples = int(data["Data"]["SerialDigitalIO"]["TimeStamp"][i][0])
            in_secs = data["Data"]["SerialDigitalIO"]["TimeStampSec"][i][0]
            marker = Onset(in_samples=in_samples, in_secs=in_secs, code=code)
            markers[code].append(marker)

        return markers


class Session(NEVMixin, ConfigSetter):
    def __init__(self, experiment, subject, config, label=None, conditions=None):
        self.uid = uuid.uuid4()
        self.experiment = experiment
        self.subject = subject
        self.config = config
        self.label = label or self.config.get("label")
        self.conditions = conditions or self.config.get("conditions")
        self.data_sources = {}
        self.ureg = self.experiment.ureg
        self._time_base = None
        self._onsets = None
        self._epoch_sets = defaultdict(lambda: EpochSet(self, epochs=[]))
        self._event_sets = defaultdict(lambda: EventSet(self, events=[]))
        self._start = None
        self._duration = None
        self.initialize_data_sources()
        self.create_epochs(self.config.get("epochs"))
        self.identities_initalized = set()

    @property
    def neurons(self):
        # do whatever you need to do to initialize neurons
        # then return self.data_identities['neurons']
        pass

    @property
    def lfp_brain_regions(self):
        # do whatever you need to do to initialize lfp brain regions
        # then return self.data_identities['lfp_brain_regions']
        pass

    @property
    def display_id(self):
        parts = [self.experiment.id, self.subject.id]
        if self.label:
            parts.append(self.label)
        if self.time_base.start_datetime:
            parts.append(str(self.time_base.start_datetime))
        else:
            parts.append(str(self.uid)[:8])
        return ":".join(parts)

    @property
    def time_base(self):
        if self._time_base is None:
            self.metadata_loader()
        return self._time_base

    @property
    def onsets(self):
        if self._onsets is None:
            self.metadata_loader()
        return self._onsets

    # TODO: it would be nice to be able to make epochs
    # into an epochs_view that could be
    # so you could access epochs.tone, epochs.pretone
    # the same way I had the idea to be able tok do subj.learning_day_1.lfp.bla etc.
    @property
    def epochs(self):
        if not self._epoch_sets:
            self.create_epochs()
        return EpochSet(
            self,
            epochs=[
                epoch
                for epoch_list in self._epoch_sets.values()
                for epoch in epoch_list
            ],
            conditions=self.conditions,
        )

    @property
    def events(self):
        if not self._event_sets:
            self.create_events()
        return EventSet(
            self,
            [event for event_list in self._event_sets.values() for event in event_list],
            conditions=self.conditions,
        )

    @property
    def start(self):
        if self._start is None:
            self.get_start_and_duration()
        return self._start

    @property
    def duration(self):
        if self._duration is None:
            self.get_start_and_duration()
        return self._duration

    def ensure_neurons(self):
        return self.ensure_identity("neuron", Neuron)

    def ensure_lfp_brain_regions(self):
        return self.ensure_identity("lfp_brain_region", LFPBrainRegion)

    def ensure_identity(self, identity_string, identity_class):
        if identity_string in self.identities_initalized:
            return

        identity_config = self.experiment.data_identity_config.get(identity_string)
        if not identity_config:
            raise ValueError("neurons were not configured for this experiment")

        # TODO: I need to figure this out.  identity_config['source'] is 'phy'
        # but self.data_sources is keyed by spike
        registry_key = identity_config["source"]
        data_sources = [
            ds
            for ds in self.data_sources.values()
            if ds.data_loader_config["registry_key"] == registry_key
        ]
        if not len(data_sources):
            return
        data_source = data_sources[0]
        components = data_source.components
        match_key = identity_config.get("match")
        if not match_key:
            for component in components:
                self.create_identity(
                    identity_class, identity_string, [component], identity_config
                )

        else:
            matched_components = []
            for di in self.subject.data_identities[identity_string]:
                match_comps = [
                    comp
                    for comp in components
                    if getattr(comp, match_key) == getattr(di, match_key)
                ]
                matched_components.extend(match_comps)
                di.add_components(matched_components)

            for component in set(components) - set(matched_components):
                self.create_identity(
                    identity_class, identity_string, [component], identity_config
                )

    def create_identity(self, identity_class, identity_string, components, config):
        identity = identity_class(components, config=config)
        self.subject.data_identities[identity_string].append(identity)

    def initialize_data_sources(self, data_sources_config=None):

        data_sources_config = data_sources_config or self.config.get("data_sources", [])

        data_sources_config = self.resolve_config(
            data_sources_config, self.experiment.data_sources_config
        )
        self.data_sources_config = data_sources_config
        for key, data_source_config in data_sources_config.items():
            self.initialize_data_source(key, data_source_config)

    def initialize_data_source(self, data_source_key, data_source_config):
        data_source_class = data_source_registry[data_source_config["registry_key"]][
            "class"
        ]
        data_source = data_source_class(self, data_source_config)
        self.data_sources[data_source_key] = data_source

    def metadata_loader(self):

        if self.config.get("nev"):
            nev_config = self.config["nev"]
            nev_data = self.load_nev(nev_config["path"])
            self._time_base = self.get_nev_time_base(nev_data)
            self._onsets = self.get_nev_onsets(nev_data)

        else:
            raise NotImplementedError("Need to add more metadata types")
        # other kinds of metadata will follow

    def create_epochs(self, epochs_config=None):
        epochs_config = self.resolve_config(
            epochs_config, self.experiment.epochs_config
        )
        for key, config in epochs_config.items():
            conditions = config.get("conditions", {})
            if "from_nev" in config:
                self.nev_epoch_config(key, config, conditions)
            elif "relative_to" in config:
                self.relative_epoch_config(key, config, conditions)

    def nev_epoch_config(self, epoch_type, epoch_config, conditions):
        code = epoch_config["code"]
        onsets = self.onsets.get(code)
        if not onsets:
            raise ValueError("Specified codes werent' found in the NEV file")

        unitful_onsets = deepcopy(onsets)

        if onsets[0].in_samples is not None:
            for onset in unitful_onsets:
                onset.in_samples *= self.ureg.raw_sample

        if onsets[0].in_secs is not None:
            for onset in unitful_onsets:
                onset.in_secs *= self.ureg.s
        else:
            for onset in unitful_onsets:
                onset.in_secs = onset.in_samples.to("s")

        duration = epoch_config["duration"] * self.ureg("second")
        self._epoch_sets[epoch_type].extend(
            [
                Epoch(
                    self,
                    onset.in_secs,
                    duration,
                    conditions=conditions,
                    config=epoch_config,
                )
                for onset in unitful_onsets
            ]
        )

    def relative_epoch_config(self, epoch_type, epoch_config, conditions):
        relative_to = epoch_config["relative_to"]
        baseline_ind = None
        bracket_matches = re.findall(r"\[([^\]]*)\]", relative_to)
        if len(bracket_matches):
            baseline_ind = int(bracket_matches[0])
            relative_to = relative_to[: relative_to.index("[")]

        target_epochs = self._epoch_sets[relative_to]
        shift = epoch_config["shift"] * self.experiment.ureg("s")
        duration = epoch_config["duration"] * self.experiment.ureg("s")
        if baseline_ind is not None:
            target_epochs = [target_epochs[baseline_ind]]
        self._epoch_sets[epoch_type].extend(
            [
                Epoch(
                    self,
                    epoch.onset - shift,
                    duration,
                    conditions=conditions,
                    config=epoch_config,
                )
                for epoch in target_epochs
            ]
        )

    def get_start_and_duration(self):
        start_sample = self.time_base.start_sample
        start_sec = (start_sample * self.ureg.raw_sample).to("s")
        self._start = start_sec

        duration_sec = self.time_base.duration_sec
        if duration_sec is not None:
            self._duration = duration_sec * self.ureg.s
        else:
            duration_sample = self.time_base.duration_sample
            if duration_sample:
                duration_sample *= self.ureg.raw_sample
                self._duration = duration_sample.to("s")

    def create_events(self):
        if self.config.get("events"):
            # this is where events that had some definition independent of epochs would live
            pass
        else:
            for epoch_type, epoch_set in self._epoch_sets.items():
                for epoch in epoch_set:
                    self._event_sets[epoch_type].extend(epoch.events)
