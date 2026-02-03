from collections import defaultdict
import h5py

from .central import ureg
from .utils import group_to_dict

class Epoch:
    
    def __init__(self, session, onset, duration, index=None, epoch_type=None):
        self.session = session
        self.onset = onset
        self.duration = duration
        self.index = index,
        self.epoch_type = epoch_type,
        self.t0 = self.onset
        self.t1 = self.onset + self.duration


class Session:
    
    def __init__(self, experiment, subject, session_config):
        self.experiment = experiment
        self.subject = subject
        self.session_config = session_config
        self.subject.sessions.append(self)
        self.epochs = defaultdict(list)
        self.create_epochs()
        
    def create_epochs(self):

        onsets_from_nev = self.session_config.get('onsets_from_nev')
        if onsets_from_nev:
            self.get_epochs_from_nev(onsets_from_nev)

    def get_epochs_from_nev(self, onsets_from_nev):
        nev_path = onsets_from_nev['nev_path']
        codes_and_onsets = self.get_codes_and_onsets(nev_path)

        for epoch_type, epoch_info in onsets_from_nev['epochs'].items():
            code = epoch_info['nev_code']
            onsets = codes_and_onsets[code]
            unitful_onsets = [onset * ureg.raw_sample for onset in onsets]
            onsets_seconds = [onset.to('second') for onset in unitful_onsets]
            duration = epoch_info['duration'] * ureg('second')
            self.epochs[epoch_type].extend(
                [Epoch(self, onset, duration) for onset in onsets_seconds])
                
                
    def get_codes_and_onsets(self, nev_path):

        with h5py.File(nev_path, 'r') as mat_file:
            data = group_to_dict(mat_file['NEV'])
            codes_and_onsets = defaultdict(list)

            for i, code in enumerate(data['Data']['SerialDigitalIO']['UnparsedData'][0]):
                onset = int(data['Data']['SerialDigitalIO']['TimeStamp'][i][0])
                codes_and_onsets[code].append(onset)

            return codes_and_onsets