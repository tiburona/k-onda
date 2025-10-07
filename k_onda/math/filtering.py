from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from copy import deepcopy


import numpy as np
from scipy.signal import (
    butter,
    cheby1,
    filtfilt,
    firwin,
    freqz,
    iirnotch,
    kaiserord,
    sosfiltfilt,
    sosfreqz,
    tf2sos,
)


@dataclass
class FilterMeta:
    method: str
    fs_eff: float
    numtaps: Optional[int] = None
    iir_order: Optional[int] = None
    iir_ripple_db: Optional[float] = None
    kaiser_atten_db: Optional[float] = None
    kaiser_tw_hz_req: Optional[float] = None
    kaiser_tw_hz_used: Optional[float] = None
    design_relaxed: Optional[bool] = None
    padlen_used: Optional[int] = None
    pass_median_db: Optional[float] = None
    stop_95_db_low: Optional[float] = None
    stop_95_db_high: Optional[float] = None


def _design_filter(fs: float, f_lo: float, f_hi: float, N: int, opts: Dict) -> Tuple[Dict, FilterMeta]:
    """
    Returns a dict describing the filter {'type': 'fir'|'sos'|'none', 'b': taps|None, 'sos': sos|None}
    and a FilterMeta filled with design details.
    """
    method = opts.get("method", "fir_hamming")  # fir_hamming | fir_kaiser | iir_butter | iir_cheby1 | iir_notch | none
    meta = FilterMeta(method=method, fs_eff=fs)

    if method == "none":
        return {"type": "none", "b": None, "sos": None}, meta

    if method == "fir_hamming":
        numtaps = int(opts.get("numtaps", round(fs)))
        if numtaps % 2 == 0:  # odd taps for bandpass
            numtaps += 1
        b = firwin(numtaps, [f_lo, f_hi], pass_zero=False, fs=fs)  # Hamming by default
        padlen = min(3 * (numtaps - 1), max(0, N - 1))
        meta.numtaps = numtaps
        meta.padlen_used = padlen

        # summarize freq response
        w, H = freqz(b, worN=8192, fs=fs)
        H2 = np.abs(H) ** 2
        pass_mask = (w >= f_lo) & (w <= f_hi)
        stop_low = (w <= max(0.0, f_lo - 10))
        stop_high = (w >= f_hi + 10)
        meta.pass_median_db = float(10 * np.log10(np.median(H2[pass_mask]) + 1e-15))
        meta.stop_95_db_low = float(10 * np.log10(np.percentile(H2[stop_low], 95) + 1e-15))
        meta.stop_95_db_high = float(10 * np.log10(np.percentile(H2[stop_high], 95) + 1e-15))

        return {"type": "fir", "b": b, "sos": None}, meta

    if method == "fir_kaiser":
        atten = float(opts.get("kaiser_atten_db", 60))
        tw = float(opts["kaiser_tw_hz"])  # required
        relax = False
        while True:
            width = tw / (fs / 2.0)
            order, beta = kaiserord(atten, width)
            if order % 2 == 1:  # even order → odd taps
                order -= 1
            numtaps = order + 1
            padlen = min(3 * (numtaps - 1), max(0, N - 1))
            if padlen < N - 1 or opts.get("allow_short", False):
                break
            # relax tw until feasible
            tw *= 1.25
            relax = True
            if tw > (f_hi - f_lo):
                break
        b = firwin(numtaps, [f_lo, f_hi], pass_zero=False, fs=fs, window=("kaiser", beta))
        meta.numtaps = numtaps
        meta.kaiser_atten_db = atten
        meta.kaiser_tw_hz_req = float(opts["kaiser_tw_hz"])
        meta.kaiser_tw_hz_used = tw
        meta.design_relaxed = relax
        meta.padlen_used = padlen
        w, H = freqz(b, worN=8192, fs=fs)
        H2 = np.abs(H) ** 2
        pass_mask = (w >= f_lo) & (w <= f_hi)
        stop_low = (w <= max(0.0, f_lo - 10))
        stop_high = (w >= f_hi + 10)
        meta.pass_median_db = float(10 * np.log10(np.median(H2[pass_mask]) + 1e-15))
        meta.stop_95_db_low = float(10 * np.log10(np.percentile(H2[stop_low], 95) + 1e-15))
        meta.stop_95_db_high = float(10 * np.log10(np.percentile(H2[stop_high], 95) + 1e-15))
        return {"type": "fir", "b": b, "sos": None}, meta

    if method == "iir_butter":
        order = int(opts.get("iir_order", 8))
        sos = butter(order, [f_lo, f_hi], btype="bandpass", fs=fs, output="sos")
        meta.iir_order = order
        # summarize
        w, H = sosfreqz(sos, worN=8192, fs=fs)
        H2 = np.abs(H) ** 2
        pass_mask = (w >= f_lo) & (w <= f_hi)
        stop_low = (w <= max(0.0, f_lo - 10))
        stop_high = (w >= f_hi + 10)
        meta.pass_median_db = float(10 * np.log10(np.median(H2[pass_mask]) + 1e-15))
        meta.stop_95_db_low = float(10 * np.log10(np.percentile(H2[stop_low], 95) + 1e-15))
        meta.stop_95_db_high = float(10 * np.log10(np.percentile(H2[stop_high], 95) + 1e-15))
        return {"type": "sos", "b": None, "sos": sos}, meta

    if method == "iir_cheby1":
        order = int(opts.get("iir_order", 6))
        rp = float(opts.get("iir_ripple_db", 0.5))  # passband ripple dB
        sos = cheby1(order, rp, [f_lo, f_hi], btype="bandpass", fs=fs, output="sos")
        meta.iir_order = order
        meta.iir_ripple_db = rp
        w, H = sosfreqz(sos, worN=8192, fs=fs)
        H2 = np.abs(H) ** 2
        pass_mask = (w >= f_lo) & (w <= f_hi)
        stop_low = (w <= max(0.0, f_lo - 10))
        stop_high = (w >= f_hi + 10)
        meta.pass_median_db = float(10 * np.log10(np.median(H2[pass_mask]) + 1e-15))
        meta.stop_95_db_low = float(10 * np.log10(np.percentile(H2[stop_low], 95) + 1e-15))
        meta.stop_95_db_high = float(10 * np.log10(np.percentile(H2[stop_high], 95) + 1e-15))
        return {"type": "sos", "b": None, "sos": sos}, meta

    if method == "iir_notch":
        # Design a narrow IIR notch filter centered at f0.
        # Use provided low/high to define bandwidth; allow overriding Q.
        f0 = 0.5 * (f_lo + f_hi)
        bw = max(1e-12, (f_hi - f_lo))
        Q_cfg = opts.get("notch_Q")
        if Q_cfg is not None:
            Q = float(Q_cfg)
        else:
            # Q = center frequency divided by bandwidth
            Q = float(f0 / bw) if bw > 0 else 30.0
        b, a = iirnotch(w0=f0, Q=Q, fs=fs)
        sos = tf2sos(b, a)
        meta.iir_order = 2  # single biquad notch
        # Summarize response: passband is outside the notch, stop is inside [f_lo, f_hi]
        w, H = sosfreqz(sos, worN=8192, fs=fs)
        H2 = np.abs(H) ** 2
        pass_mask = (w <= max(0.0, f_lo - 10)) | (w >= f_hi + 10)
        stop_band = (w >= f_lo) & (w <= f_hi)
        if np.any(pass_mask):
            meta.pass_median_db = float(10 * np.log10(np.median(H2[pass_mask]) + 1e-15))
        if np.any(stop_band):
            stop_db = float(10 * np.log10(np.percentile(H2[stop_band], 95) + 1e-15))
            meta.stop_95_db_low = stop_db
            meta.stop_95_db_high = stop_db
        return {"type": "sos", "b": None, "sos": sos}, meta

    raise ValueError(f"Unknown method: {method}")


class Filter:
    """
    Wrapper providing filter design + application.

    Expected config keys:
      - fs: sampling rate (Hz), required
      - low: low cutoff (Hz), required
      - high: high cutoff (Hz), required
      - method: 'fir_hamming' | 'fir_kaiser' | 'iir_butter' | 'iir_cheby1' | 'iir_notch' | 'none'
      - Optional method-specific params as used in _design_filter

    If config is falsy, acts as identity.
    """

    def __init__(self, config: Optional[Dict]):  # finish this init function
        # Identity / no-op
        if not config:
            self._kind = "none"
            self._b = None
            self._sos = None
            self.meta = FilterMeta(method="none", fs_eff=float('nan'))
            return

        cfg = dict(config)
        fs = cfg.get("fs")
        f_lo = cfg.get("low")
        f_hi = cfg.get("high")
      
        if fs is None or f_lo is None or f_hi is None:
            raise ValueError("Filter requires 'fs' and 'low', and 'high' in config")

        # N may be optionally provided to compute a safe padlen; if not, we will
        # compute a safe runtime padlen based on input length.
        N = int(cfg.get("N", 0))

        design, meta = _design_filter(float(fs), float(f_lo), float(f_hi), int(N), cfg)
        self.meta = meta
        self._kind = design["type"]
        self._b = design.get("b")
        self._sos = design.get("sos")

    def __call__(self, data: np.ndarray) -> np.ndarray:
        x = np.asarray(data)
        if x.ndim != 1:
            raise ValueError("Filter expects a 1D array")
        if self._kind == "none":
            return x
        if self._kind == "fir":
            # Safe padlen: follow scipy's default but cap to len(x)-1
            numtaps = len(self._b)
            padlen = min(3 * (numtaps - 1), max(0, x.size - 1))
            return filtfilt(self._b, [1.0], x, padlen=padlen)
        if self._kind == "sos":
            # sosfiltfilt chooses padlen internally; rely on default
            return sosfiltfilt(self._sos, x)
        raise RuntimeError(f"Unknown filter kind: {self._kind}")



class FilterMixin:
    """
    Mixin that builds a Filter from defaults + user overrides at calc_opts['filters'][section].
    Requires the host class to provide:
      - self.calc_opts: dict
      - self.get_or_create_filter(cfg: dict) -> Filter
      - self.shared_filters: dict (optional, only if you use 'alias')
    """

    def make_filter(
        self,
        default_cfg: Dict,
        section: str,
        *,
        extra: Optional[Dict] = None,
        alias: Optional[Tuple] = None,
    ):
        """
        default_cfg: baseline design (fs, low, high, method, etc.)
        section: key under calc_opts['filters'] to pull user overrides from
        extra: programmatic overrides to apply after defaults but before user overrides
        alias: optional cache alias key, e.g. ('notch', default_cfg['fs'])
        """
        cfg = deepcopy(default_cfg)

        if extra:
            cfg.update(extra)

        user_overrides = (
            (self.calc_opts or {}).get('filters', {}).get(section, {})  # safe if missing
        )
        cfg.update(user_overrides)

        flt = self.get_or_create_filter(cfg)

        if alias is not None:
            # convenience alias, does not affect the canonical hashed key
            self.shared_filters[alias] = flt

        return flt, cfg  

    def visualize_filter(
        self,
        flt: "Filter",
        *,
        zoom_range: Optional[Tuple[float, float]] = None,  # e.g., (6.0, 14.0)
        worN: int = 8192,
        db: bool = True,
        annotate: bool = True,
    ):
        """
        Plot magnitude response of a designed Filter (FIR/IIR).
        - zoom_range: (f_lo, f_hi) in Hz to show a zoom panel in addition to the full view.
        - worN: frequency grid size for response.
        - db: plot in dB (20*log10|H|) if True, otherwise linear gain.
        - annotate: add a small text box with design metadata.

        Returns
        -------
        (fig, axes)
        axes is a single Axes when zoom_range is None, else a tuple (ax_full, ax_zoom).
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.signal import freqz, sosfreqz

        if flt._kind == "none":
            raise ValueError("Cannot visualize a 'none' filter.")

        fs = float(flt.meta.fs_eff) if np.isfinite(getattr(flt.meta, "fs_eff", np.nan)) else None
        if fs is None:
            raise ValueError("Filter.meta.fs_eff is missing; cannot compute frequency response.")

        # Frequency response
        if flt._kind == "fir":
            w, H = freqz(flt._b, worN=worN, fs=fs)
        elif flt._kind == "sos":
            w, H = sosfreqz(flt._sos, worN=worN, fs=fs)
        else:
            raise RuntimeError(f"Unknown filter kind: {flt._kind}")

        mag = np.abs(H)
        y = 20.0 * np.log10(np.maximum(mag, 1e-12)) if db else mag

        # Figure layout: one panel or full+zoom
        if zoom_range is None:
            fig, ax = plt.subplots(figsize=(7.5, 4.0))
            ax.plot(w, y, lw=1.6)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Magnitude (dB)" if db else "Gain")
            ax.set_title(f"Filter response: {flt.meta.method}")
            ax.grid(True, alpha=0.3)
            axes = ax
        else:
            fz0, fz1 = map(float, zoom_range)
            mask = (w >= min(fz0, fz1)) & (w <= max(fz0, fz1))

            fig, (ax_full, ax_zoom) = plt.subplots(
                2, 1, figsize=(7.5, 6.0), height_ratios=[2, 1], sharey=True
            )
            # Full
            ax_full.plot(w, y, lw=1.6)
            ax_full.set_xlabel("Frequency (Hz)")
            ax_full.set_ylabel("Magnitude (dB)" if db else "Gain")
            ax_full.set_title(f"Filter response: {flt.meta.method} (full)")
            ax_full.grid(True, alpha=0.3)

            # Zoom
            ax_zoom.plot(w[mask], y[mask], lw=1.8)
            ax_zoom.set_xlim(min(fz0, fz1), max(fz0, fz1))
            ax_zoom.set_xlabel("Frequency (Hz)")
            ax_zoom.set_ylabel("Magnitude (dB)" if db else "Gain")
            ax_zoom.set_title(f"Zoom: {min(fz0, fz1):.3g}–{max(fz0, fz1):.3g} Hz")
            ax_zoom.grid(True, alpha=0.3)

            axes = (ax_full, ax_zoom)

        if annotate:
            # Build a compact metadata blurb (only fields that exist)
            meta = flt.meta
            lines = [f"method: {meta.method}", f"fs: {getattr(meta, 'fs_eff', 'n/a'):g} Hz"]
            if getattr(meta, "numtaps", None):
                lines.append(f"numtaps: {meta.numtaps}")
            if getattr(meta, "iir_order", None):
                lines.append(f"iir_order: {meta.iir_order}")
            if getattr(meta, "iir_ripple_db", None):
                lines.append(f"rip(dB): {meta.iir_ripple_db}")
            if getattr(meta, "kaiser_atten_db", None):
                lines.append(f"kaiser_attn(dB): {meta.kaiser_atten_db}")
            if getattr(meta, "kaiser_tw_hz_used", None):
                lines.append(f"kaiser_tw_used: {meta.kaiser_tw_hz_used:g} Hz")
            if getattr(meta, "padlen_used", None):
                lines.append(f"padlen_used: {meta.padlen_used}")
            if getattr(meta, "pass_median_db", None):
                lines.append(f"pass_med(dB): {meta.pass_median_db:.1f}")
            if getattr(meta, "stop_95_db_low", None):
                lines.append(f"stop95 low(dB): {meta.stop_95_db_low:.1f}")
            if getattr(meta, "stop_95_db_high", None):
                lines.append(f"stop95 high(dB): {meta.stop_95_db_high:.1f}")

            txt = "\n".join(lines)
            # Choose the first axis for the note
            ax0 = axes if not isinstance(axes, tuple) else axes[0]
            ax0.text(
                0.99, 0.02, txt,
                transform=ax0.transAxes,
                ha="right", va="bottom",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, linewidth=0.5),
            )

        # Make sure a window pops up when running outside notebooks
        import matplotlib.pyplot as plt
        plt.tight_layout()
        plt.show()

        return fig, axes

