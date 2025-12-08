
from .coherence import (psd, cross_spectral_density, 
    msc_from_spectra, fisher_z_from_coherence, fisher_z_from_msc, 
    back_transform_fisher_z)
from .correlation import normalized_xcorr, pearson_xcorr
from .filtering import Filter, FilterMixin
from .mrl import compute_mrl, compute_phase
from .hilbert import apply_hilbert_to_padded_data
from .misc import pool
from .rates import calc_hist, calc_rates
