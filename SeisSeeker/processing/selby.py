"""Selby-inspired detector utilities.

This module provides a practical, diagonal-noise approximation to the
multiple-filter generalized F detector described by Selby (2013).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt


def build_selby_filter_centres(
    freqmin: float,
    freqmax: float,
    step_hz: float,
    half_width_hz: float,
) -> np.ndarray:
    """Build a set of overlapping filter centres within a band."""
    if freqmin <= 0:
        raise ValueError("freqmin must be positive for the Selby filter bank.")
    if freqmax <= freqmin:
        raise ValueError("freqmax must be greater than freqmin.")
    if step_hz <= 0:
        raise ValueError("step_hz must be positive.")
    if half_width_hz <= 0:
        raise ValueError("half_width_hz must be positive.")

    lower = freqmin + half_width_hz
    upper = freqmax - half_width_hz
    if lower > upper:
        return np.array([(freqmin + freqmax) / 2.0], dtype=float)

    centres = np.arange(lower, upper + (0.5 * step_hz), step_hz, dtype=float)
    if len(centres) == 0:
        centres = np.array([(freqmin + freqmax) / 2.0], dtype=float)
    return centres


def rolling_noise_floor(window_powers: np.ndarray, noise_window_len: int) -> np.ndarray:
    """Estimate a trailing rolling median noise floor for each station."""
    if noise_window_len <= 0:
        raise ValueError("noise_window_len must be positive.")
    if window_powers.ndim != 2:
        raise ValueError("window_powers must have shape (n_windows, n_stations).")

    n_windows, n_stations = window_powers.shape
    noise_floor = np.zeros_like(window_powers, dtype=float)
    eps = np.finfo(float).eps

    for station_idx in range(n_stations):
        station_series = np.maximum(window_powers[:, station_idx], eps)
        for win_idx in range(n_windows):
            start = max(0, win_idx - noise_window_len + 1)
            noise_floor[win_idx, station_idx] = np.median(
                station_series[start : win_idx + 1]
            )

    return np.maximum(noise_floor, eps)


def _compute_window_powers(
    analytic_data: np.ndarray,
    win_len_samples: int,
    step_samples: int,
    n_windows: int,
) -> np.ndarray:
    """Calculate mean analytic power in each processing window."""
    n_stations = analytic_data.shape[0]
    powers = np.zeros((n_windows, n_stations), dtype=float)
    for win_idx in range(n_windows):
        start = win_idx * step_samples
        end = start + win_len_samples
        powers[win_idx, :] = np.mean(np.abs(analytic_data[:, start:end]) ** 2, axis=1)
    return powers


def compute_selby_score_cube(
    data: np.ndarray,
    fs: float,
    xx: np.ndarray,
    yy: np.ndarray,
    min_sl: float,
    max_sl: float,
    n_sl: int,
    min_baz: float,
    max_baz: float,
    n_baz: int,
    win_len_s: float,
    win_step_inc_s: float,
    freqmin: float,
    freqmax: float,
    filter_step_hz: float,
    filter_half_width_hz: float,
    noise_window_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a Selby-inspired F-score cube over time/slowness/back-azimuth."""
    if data.ndim != 2:
        raise ValueError("data must have shape (n_stations, n_samples).")

    n_stations, n_samples = data.shape
    win_len_samples = int(round(win_len_s * fs))
    step_samples = int(round(win_step_inc_s * fs))
    if win_len_samples <= 0 or step_samples <= 0:
        raise ValueError("Window length and step must produce positive sample counts.")
    if n_samples < win_len_samples:
        return np.zeros((0, n_sl, n_baz), dtype=float), np.array([], dtype=float)

    n_windows = 1 + int((n_samples - win_len_samples) / step_samples)
    if n_windows <= 0:
        return np.zeros((0, n_sl, n_baz), dtype=float), np.array([], dtype=float)

    ur = np.linspace(min_sl, max_sl, n_sl)
    utheta = np.linspace(min_baz, max_baz, n_baz)
    utheta_rad = np.deg2rad(utheta)
    timeshifts = np.zeros((n_stations, n_sl, n_baz), dtype=float)
    for ir in range(n_sl):
        for itheta in range(n_baz):
            timeshifts[:, ir, itheta] = (
                xx * ur[ir] * np.sin(utheta_rad[itheta])
                + yy * ur[ir] * np.cos(utheta_rad[itheta])
            )
    timeshifts = timeshifts.reshape(n_stations, -1)

    filter_centres = build_selby_filter_centres(
        freqmin, freqmax, filter_step_hz, filter_half_width_hz
    )
    scores = np.zeros((n_windows, n_sl * n_baz), dtype=float)
    valid_bands = 0
    eps = np.finfo(float).eps
    noise_window_len = max(1, int(round(noise_window_s / win_step_inc_s)))

    nyquist = 0.5 * fs
    for centre_freq in filter_centres:
        low = max(freqmin, centre_freq - filter_half_width_hz)
        high = min(freqmax, centre_freq + filter_half_width_hz, nyquist * 0.999)
        if high <= low:
            continue

        sos = butter(4, [low, high], btype="bandpass", output="sos", fs=fs)
        filtered = sosfiltfilt(sos, data, axis=-1)
        analytic = hilbert(filtered, axis=-1).astype(np.complex128)
        window_powers = _compute_window_powers(
            analytic, win_len_samples, step_samples, n_windows
        )
        noise_floor = rolling_noise_floor(window_powers, noise_window_len)
        steering = np.exp(
            -1j * 2.0 * np.pi * centre_freq * timeshifts.T
        )  # (n_candidates, n_stations)

        for win_idx in range(n_windows):
            start = win_idx * step_samples
            end = start + win_len_samples
            window_data = analytic[:, start:end] / np.sqrt(
                noise_floor[win_idx, :, np.newaxis] + eps
            )
            cov = np.matmul(np.conj(window_data), window_data.T) / float(
                win_len_samples
            )
            tracepower = np.real(np.trace(cov))
            if tracepower <= eps:
                continue

            steering_cov = np.matmul(steering, cov)
            beampower = (
                np.sum(steering_cov * np.conj(steering), axis=1).real
                / float(n_stations**2)
            )
            denom = np.maximum(tracepower - (n_stations * beampower), eps)
            band_scores = ((n_stations - 1) * n_stations * beampower) / denom
            scores[win_idx, :] += band_scores

        valid_bands += 1

    if valid_bands > 0:
        scores /= float(valid_bands)

    score_cube = scores.reshape(n_windows, n_sl, n_baz)
    centres = np.arange(
        win_step_inc_s / 2.0,
        (n_windows * win_step_inc_s) + (win_step_inc_s / 2.0),
        win_step_inc_s,
    )
    if len(centres) > n_windows:
        centres = centres[:n_windows]
    return score_cube, centres
