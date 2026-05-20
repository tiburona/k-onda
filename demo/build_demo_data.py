import numpy as np
from scipy.signal import butter, sosfiltfilt
import math
import os
from pathlib import Path

from k_onda.model import Experiment

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)

DATA_DIR = REPO_ROOT / "demo" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)


def bandpass_noise(rng, n, fs, low, high, order=4):
    white = rng.normal(size=n)
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    y = sosfiltfilt(sos, white)
    return y / y.std()

def raised_cosine_window(t, start, stop, ramp):
    env = np.zeros_like(t, dtype=float)
    up = (t >= start) & (t < start + ramp)
    mid = (t >= start + ramp) & (t <= stop - ramp)
    down = (t > stop - ramp) & (t <= stop)

    env[mid] = 1.0
    env[up] = 0.5 - 0.5 * np.cos(np.pi * (t[up] - start) / ramp)
    env[down] = 0.5 + 0.5 * np.cos(np.pi * (t[down] - (stop - ramp)) / ramp)

    return env


def generate_animal_lfp(animal):

    fs = 500
    duration = 120
    t = np.arange(0, duration, 1 / fs)
    n = len(t)

    broadband = 0.6 * rng.normal(size=n)
    theta = bandpass_noise(rng, n, fs, 4, 8)

    theta_event = raised_cosine_window(t, start=10, stop=22, ramp=2)

    theta_amp_low = 0.8
    theta_amp_high = theta_amp_low * np.sqrt(2)
    theta_amp = theta_amp_low + (theta_amp_high - theta_amp_low) * theta_event

    lfp = (broadband + theta_amp * theta)[:, None]

    np.save(DATA_DIR / f"{animal}_lfp.npy", lfp)
   

def random_spikes(rate, start, stop):
    spikes = []
    time = start
    while time < stop:
        time_to_next = -math.log(1.0 - rng.random()) / rate
        time += time_to_next
        if time < stop:
            spikes.append(time)
    return spikes


def gaussian(t, center, sigma):
    return np.exp(-0.5 * ((t - center) / sigma) ** 2)


def random_waveforms( 
        trough_amp, 
        rebound_amp, 
        trough_sigma, 
        rebound_sigma, 
        trough_time, 
        rebound_time,
        n_spikes,
        n_samples, 
        fs
        ):
    
    waveforms = []

    t = np.arange(n_samples) / fs

    for _ in range(n_spikes):
        spike_trough_amp = trough_amp + rng.normal(0, 0.2 * trough_amp)
        spike_rebound_amp = rebound_amp + rng.normal(0, 0.2 * rebound_amp)
        spike_trough_sigma = trough_sigma + rng.normal(0, 0.2 * trough_sigma)
        spike_rebound_sigma = rebound_sigma + rng.normal(0, 0.2 * rebound_sigma)
        spike_trough_time = trough_time + rng.normal(0, 0.2 * trough_time)
        spike_rebound_time = rebound_time + rng.normal(0, 0.2 * rebound_time)

        while spike_rebound_time <= spike_trough_time:
            spike_rebound_time = rebound_time + rng.normal(0, 0.2 * rebound_time)
        
        waveform = (
            -spike_trough_amp * gaussian(t, spike_trough_time, spike_trough_sigma)
            + spike_rebound_amp * gaussian(t, spike_rebound_time, spike_rebound_sigma)
        )

        waveforms.append(waveform)

    return waveforms


def fwhm_to_sigma(fwhm):
    return fwhm / (2 * np.sqrt(2 * np.log(2)))


fs = 30000.
trough_time = 0.001
rebound_time = 0.002
n_samples = 200

PN_rate = 2
PN_fwhm = 0.00055
IN_rate = 12
IN_fwhm = 0.00025


def generate_animal_neurons(animal):
    rng = np.random.default_rng(42)
    n_PNs = 4
    n_INs = 8

    PN_clusters = []
    IN_clusters = []
    PN_waveforms = []
    IN_waveforms = []
    PN_spikes = []
    IN_spikes = []

    for i in range(n_PNs):
        rate = PN_rate + rng.normal(0, PN_rate/5)
        spikes = random_spikes(rate, 0, 60)
        waveforms = random_waveforms(
            trough_amp=100, 
            trough_sigma=fwhm_to_sigma(PN_fwhm), 
            rebound_amp=20, 
            rebound_sigma=fwhm_to_sigma(.5 * PN_fwhm), 
            trough_time=trough_time,
            rebound_time=rebound_time,
            n_spikes=len(spikes),
            n_samples=n_samples,
            fs=fs)
        PN_clusters.append(np.repeat(i, len(spikes)))
        PN_waveforms.append(waveforms) 
        PN_spikes.append(spikes) 
    
      
    for i in range(n_INs):
        rate = IN_rate + rng.normal(0, IN_rate/5)
        spikes = random_spikes(rate, 0, 60)
        waveforms = random_waveforms(
            trough_amp=100, 
            trough_sigma=fwhm_to_sigma(IN_fwhm), 
            rebound_amp=20, 
            rebound_sigma=fwhm_to_sigma(.5 * IN_fwhm), 
            trough_time=trough_time,
            rebound_time=rebound_time,
            n_spikes=len(spikes),
            n_samples=n_samples,
            fs=fs)
        IN_clusters.append(np.repeat(i + n_PNs, len(spikes)))
        IN_waveforms.append(waveforms) 
        IN_spikes.append(spikes) 
        
    clusters = np.concatenate(PN_clusters + IN_clusters)
    spike_times = np.concatenate(PN_spikes + IN_spikes)
    waveforms = np.concatenate(PN_waveforms + IN_waveforms)
          
    np.savez_compressed(
        DATA_DIR / f"{animal}_spikes.npz", 
        clusters=clusters, 
        spike_times=spike_times, 
        waveforms=waveforms
        )


os.makedirs("demo/data", exist_ok=True)

animals = ["animal_1", "animal_2", "animal_3", "animal_4", "animal_5", "animal_6"]

for animal in animals:
    generate_animal_neurons(animal)
    generate_animal_lfp(animal)


experiment = Experiment.from_config(
    "demo", 
    global_config="demo/demo_config.yaml").initialize()

















        

        

       



