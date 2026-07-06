#!/usr/bin/python
# -----------------------------------------------------------------------------------------------------------------------------------------

# Script Description:
# Script to perform earthquake detection using array processing methods.

# Created by Tom Hudson, 10th August 2022

# -----------------------------------------------------------------------------------------------------------------------------------------

# Import neccessary modules:
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os
import obspy
import datetime
from scipy.signal import find_peaks, hilbert
try:
    from numba import jit, objmode, prange
except ImportError:  # pragma: no cover - exercised indirectly in test envs
    from contextlib import contextmanager

    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    @contextmanager
    def objmode(**kwargs):
        yield

    def prange(*args):
        return range(*args)
import gc
import logging

import time
import glob
import pickle
from SeisSeeker.processing import lookup_table_manager, location, selby

logger = logging.getLogger(__name__)

# GLobal constants:
MAD_SCALE = 1.4826  # Scale factor to convert MAD to std. dev.


# ---------- Define main functions ------------
class CustomError(Exception):
    pass


def flatten_list(list_to_flatten):
    return [item for sublist in list_to_flatten for item in sublist]


def xy_to_rtheta(x, y):
    """x,y to r,theta, where x,y in East and North directions.
    Theta is in degrees from N."""
    r = np.sqrt(x**2 + y**2)
    theta = np.rad2deg(np.arctan2(x, y))
    try:
        theta[theta < 0] = theta[theta < 0] + 360
    except TypeError:
        theta = theta + 360
    return r, theta


def _ensure_score_columns(t_series_df):
    """Backfill generic score columns for legacy detector outputs."""
    if "score" not in t_series_df.columns:
        t_series_df["score"] = t_series_df["power"].values
    if "statistic" not in t_series_df.columns:
        t_series_df["statistic"] = "beam_power"
    return t_series_df


@jit(nopython=True, parallel=True)
def _fast_freq_domain_array_proc(
    data,
    min_sl,
    max_sl,
    n_sl,
    min_baz,
    max_baz,
    n_baz,
    fs,
    target_freqs,
    xx,
    yy,
    n_stations,
    n_t_samp,
    remove_autocorr,
):
    """
    Performs array processing using method inspired by Bowden et al. (2021).
    Performs array processing in polar coordinates.
    Designed to be wrapped using Numba to improve performance.

    Parameters:
    ----------
    data : np.ndarray
        3D numpy array of data to process. Shape must be
        (n_windows, n_stations, n_t_samp).
    min_sl : float
        Minimum slowness to analyse for, in s/km.
    max_sl : float
        Maximum slowness to analyse for, in s/km.
    n_sl : int
        Number of slowness values to analyse between min_sl and max_sl.
    min_baz : float
        Minimum back-azimuth, in degrees.
    max_baz : float
        Maximum back-azimuth, in degrees.
    n_baz : int
        Number of back-azimuth values to analyse between min_baz and max_baz.
    fs : float
        Sampling frequency of the data, in Hz.
    target_freqs : list
        List of target frequencies to analyse, in Hz.
    xx : np.ndarray
        1D numpy array of x-coordinates of station locations, in km.
    yy : np.ndarray
        1D numpy array of y-coordinates of station locations, in km.
    n_stations : int
        Number of stations in the array.
    n_t_samp : int
        Number of time samples in the data.
    remove_autocorr : bool
        Whether to remove autocorrelations from the data.

    Returns:
    ----------
    Pfreq_all : np.ndarray
        4D numpy array of processed data. Shape will be
        (n_windows, len(target_freqs), n_sl, n_baz).
    """
    # Define grid of slownesses:
    ur = np.linspace(min_sl, max_sl, n_sl)
    utheta = np.linspace(min_baz, max_baz, n_baz)
    utheta_rad = np.deg2rad(utheta)

    # # Compute time-shifts once:
    tlib = np.zeros((n_stations, n_sl, n_baz), dtype=np.complex128)
    # r, theta as this is polar coord system:
    for ir in range(n_sl):
        for itheta in range(n_baz):
            tlib[:, ir, itheta] = xx * ur[ir] * np.sin(utheta_rad[itheta]) + yy * ur[
                ir
            ] * np.cos(utheta_rad[itheta])

    # Vectorized computation of time-shifts for all stations, slowness, and back-azimuth
    # xx and yy are (n_stations,) arrays, ur is (n_sl,), utheta_rad is (n_baz,)
    # We want tlib shape (n_stations, n_sl, n_baz)

    # xx_ = xx[:, np.newaxis, np.newaxis]  # shape (n_stations, 1, 1)
    # yy_ = yy[:, np.newaxis, np.newaxis]  # shape (n_stations, 1, 1)
    # ur_ = ur[np.newaxis, :, np.newaxis]  # shape (1, n_sl, 1)
    # utheta_rad_ = utheta_rad[np.newaxis, np.newaxis, :]  # shape (1, 1, n_baz)
    # tlib_tmp = xx_ * ur_ * np.sin(utheta_rad_) + yy_ * ur_ * np.cos(utheta_rad_)
    # recast array as complex128
    # tlib = tlib_tmp.astype(np.complex128)
    Pfreq_all = np.zeros(
        (data.shape[0], len(target_freqs), n_sl, n_baz), dtype=np.complex128
    )
    for win_idx in prange(data.shape[0]):
        nfft = int(2 ** np.ceil(np.log2(n_t_samp)))
        Pxx_all = np.zeros((int(nfft / 2) + 1, n_stations), dtype=np.complex128)
        dt = 1.0 / fs
        xf = np.linspace(0.0, 1.0 / (2.0 * dt), int(nfft / 2) + 1)
        # Calculate power spectra for all stations:
        for sta_idx in range(n_stations):
            with objmode(Pxx_curr="complex128[:]"):
                Pxx_curr = np.fft.rfft(data[win_idx, sta_idx, :], n=nfft)
            Pxx_all[:, sta_idx] = Pxx_curr

        Pfreq = np.zeros((len(target_freqs), n_sl, n_baz), dtype=np.complex128)
        for ii in range(len(target_freqs)):
            target_f = target_freqs[ii]
            curr_f_idx = (np.abs(xf - target_f)).argmin()

            Rxx = np.zeros((n_stations, n_stations), dtype=np.complex128)
            for i1 in range(n_stations):
                for i2 in range(n_stations):
                    if remove_autocorr:
                        if i1 != i2:
                            Rxx[i1, i2] = (
                                np.conj(Pxx_all[curr_f_idx, i1])
                                * Pxx_all[curr_f_idx, i2]
                            )
                        else:
                            Rxx[i1, i2] = 0
                    else:
                        Rxx[i1, i2] = (
                            np.conj(Pxx_all[curr_f_idx, i1]) * Pxx_all[curr_f_idx, i2]
                        )

            for ir in range(n_sl):
                for itheta in range(n_baz):
                    timeshifts = tlib[:, ir, itheta]
                    a = np.exp(-1j * 2 * np.pi * target_f * timeshifts)
                    aconj = np.conj(a)
                    Pfreq[ii, ir, itheta] = np.dot(np.dot(aconj, Rxx), a)

        Pfreq_all[win_idx, :, :, :] = Pfreq

    return Pfreq_all


# @jit(nopython=True, parallel=True)#, nogil=True)
def _phase_associator_core_worker(
    peaks_Z,
    peaks_hor,
    bazis_Z,
    bazis_hor,
    bazi_tol,
    t_Z_secs_after_start,
    t_hor_secs_after_start,
    max_phase_sep_s,
    min_phase_sep_s,
):
    """Function to do the heavy lifting of the phase association."""
    # Specify data stores:
    Z_hor_phase_pair_idxs = []

    # Loop over phases, seeing if they meet phase association criteria:
    for i in range(len(peaks_Z)):
        if i % 1000 == 0:
            print(i, "/", len(peaks_Z))
        curr_peak_Z_idx = peaks_Z[i]
        for j in range(len(peaks_hor)):
            curr_peak_hor_idx = peaks_hor[j]
            # i. Check if phase arrivals are within specified time limits:
            curr_t_phase_diff = (
                t_hor_secs_after_start[curr_peak_hor_idx]
                - t_Z_secs_after_start[curr_peak_Z_idx]
            )
            if (
                (curr_t_phase_diff > 0)
                and (curr_t_phase_diff <= max_phase_sep_s)
                and (curr_t_phase_diff >= min_phase_sep_s)
            ):
                # calc bazi diff between Z and H
                bazi_diff = min(
                    np.abs(bazis_Z[i] - bazis_hor[j]),
                    360.0 - np.abs(bazis_Z[i] - bazis_hor[j]),
                )
                # ii. Check if bazis for Z and horizontals current pick match:
                if bazi_diff < bazi_tol:
                    match = [curr_peak_Z_idx, curr_peak_hor_idx]
                    # And associate phases and create event data if a match is found:
                    # Append pair idxs to data store:
                    Z_hor_phase_pair_idxs.append(match)

    return Z_hor_phase_pair_idxs


def _phase_associator(
    t_series_df_Z,
    t_series_df_hor,
    peaks_Z,
    peaks_hor,
    bazi_tol,
    filt_phase_assoc_by_max_power,
    max_phase_sep_s,
    min_phase_sep_s,
    min_event_sep_s,
    verbosity=0,
):
    """
    Function to perform phase association for numba implementation.
    """
    # Find back-azimuths associated with phase picks:
    bazis_Z = t_series_df_Z["back_azi"].values[peaks_Z]
    bazis_hor = t_series_df_hor["back_azi"].values[peaks_hor]

    # -------------------------------------------------------------------
    # Perform core phase association:
    # Prep. data for numba format:
    if verbosity > 1:
        logger.info("Pre-processing time-series")
    t_Z_secs_after_start = np.array(t_series_df_Z["t"] - t_series_df_Z["t"][0])
    t_hor_secs_after_start = np.array(t_series_df_hor["t"] - t_series_df_hor["t"][0])

    if verbosity > 1:
        logger.info("Performing phase association")
    Z_hor_phase_pair_idxs = _phase_associator_core_worker(
        peaks_Z,
        peaks_hor,
        bazis_Z,
        bazis_hor,
        bazi_tol,
        t_Z_secs_after_start,
        t_hor_secs_after_start,
        max_phase_sep_s,
        min_phase_sep_s,
    )
    # Organise outputs into useful form:
    if verbosity > 1:
        logger.info("Writing events")

    curr_events = {
        "t1": [],
        "t2": [],
        "pow1": [],
        "pow2": [],
        "slow1": [],
        "slow2": [],
        "bazi1": [],
        "bazi2": [],
    }

    for event_idx in range(len(Z_hor_phase_pair_idxs)):
        curr_peak_Z_idx = Z_hor_phase_pair_idxs[event_idx][0]
        curr_peak_hor_idx = Z_hor_phase_pair_idxs[event_idx][1]
        curr_events["t1"].append(t_series_df_Z["t"][curr_peak_Z_idx])
        curr_events["t2"].append(t_series_df_hor["t"][curr_peak_hor_idx])
        curr_events["pow1"].append(t_series_df_Z["power"][curr_peak_Z_idx])
        curr_events["pow2"].append(t_series_df_hor["power"][curr_peak_hor_idx])
        curr_events["slow1"].append(t_series_df_Z["slowness"][curr_peak_Z_idx])
        curr_events["slow2"].append(t_series_df_hor["slowness"][curr_peak_hor_idx])
        curr_events["bazi1"].append(t_series_df_Z["back_azi"][curr_peak_Z_idx])
        curr_events["bazi2"].append(t_series_df_hor["back_azi"][curr_peak_hor_idx])

    events_df = pd.DataFrame(curr_events)
    # And tidy:
    del t_Z_secs_after_start, t_hor_secs_after_start, Z_hor_phase_pair_idxs
    gc.collect()
    # -------------------------------------------------------------------

    # And filter events to only output events with max. power within the max. phase window,
    # if speficifed by user:
    if filt_phase_assoc_by_max_power:
        # Only process if found some events:
        if len(events_df) > 0:
            # Calculate max power of P and S for each potential event:
            # events_overall_powers = events_df['pow1'].values + events_df['pow2'].values
            # Define datastores:
            filt_events_lst = []
            # And loop over events, selecting only max. power events:
            tmp_count = 0
            for _, row in events_df.iterrows():
                tmp_count += 1
                if tmp_count == 1:
                    tmp_lst = []
                    tmp_lst.append(row)
                else:

                    # Append event if phase within minimum event separation:
                    if (
                        obspy.UTCDateTime(row["t1"]) - obspy.UTCDateTime(tmp_lst[0].t1)
                        < min_event_sep_s
                    ):
                        # Append event to compare:
                        tmp_lst.append(row)
                    else:
                        # Find best event from previous events:
                        max_power_event = _find_max_power_event(tmp_lst)
                        filt_events_lst.append(max_power_event)

                        # And start acrewing new events:
                        tmp_lst = []
                        tmp_lst.append(row)
            # And calculate highest power event for final window:
            max_power_event = _find_max_power_event(tmp_lst)
            filt_events_lst.append(max_power_event)
            # Now make new DataFrame
            filt_events_df = pd.DataFrame(filt_events_lst)
            # And sort indices:
            filt_events_df.reset_index(drop=True, inplace=True)

            # And remove duplicate S pick associations:
            # (using same max. power method)
            # Append summed powers, for sorting:
            sum_pows = filt_events_df["pow1"].values + filt_events_df["pow2"].values
            filt_events_df["sum_pows"] = sum_pows
            # Remove t2 duplicates, keep highest summed power:
            filt_events_df.sort_values("sum_pows", inplace=True)
            filt_events_df.drop_duplicates(subset="t2", keep="last", inplace=True)
            # And remove sum_pows column:
            filt_events_df.drop(columns=["sum_pows"], inplace=True)

            # And output df:
            events_df = filt_events_df.copy()
            del filt_events_df
            gc.collect()

    return events_df


def _find_max_power_event(events):
    """
    Find the maximum power event from a list of events

    Parameters:
    ----------
    events : list
        list (of Dataframe Rows) of event

    """
    pow1_tmp = np.array([event.pow1 for event in events])
    pow2_tmp = np.array([event.pow2 for event in events])
    combined_pows_tmp = pow1_tmp + pow2_tmp
    max_power_idx = np.argmax(combined_pows_tmp)
    max_power_event = events[max_power_idx]
    return max_power_event


def _submit_parallel_fast_freq_domain_array_proc(
    procnum,
    return_dict_Pfreq_all,
    data_curr_run,
    max_sl,
    fs,
    target_freqs,
    xx,
    yy,
    n_stations,
    n_t_samp,
    remove_autocorr,
):
    """Function to submit parallel runs of _fast_freq_domain_array_proc() function."""
    # Run function:
    Pfreq_all_curr_run = _fast_freq_domain_array_proc(
        data_curr_run,
        max_sl,
        fs,
        target_freqs,
        xx,
        yy,
        n_stations,
        n_t_samp,
        remove_autocorr,
    )
    # And return data
    return_dict_Pfreq_all[procnum] = Pfreq_all_curr_run


def _calc_time_shift_from_array_cent(slow, bazi, x_rec, y_rec):
    """Calculates time shift of signal at receiver from array centre for stacking data.
    Note: All distances and velocities use km unless otherwise specified.

    Parameters
    ----------
    slow : float
        Slowness of arrival, in s/km.
    bazi : float
        Back azimuth of arrival in degrees from .
    x_rec :
        Location of receiver in km East.
    y_rec
        Location of receiver in km North.
    """
    # Calculate time-shift for receiver:
    # (in polar coord system, for consistency)
    bazi_rad = np.deg2rad(bazi)
    time_shift_curr = x_rec * slow * np.sin(bazi_rad) + y_rec * slow * np.cos(
        bazi_rad
    )  # (distance x slowness = distance / velocity = time)
    return time_shift_curr


def _create_stacked_data_st(st, Z_all, N_all, E_all):
    """Function to create stacked data st."""

    composite_st = obspy.Stream()
    # For Z stacked:
    tr = st[0].copy()
    tr.stats.station = "STACK"
    tr.stats.channel = st[0].stats.channel[0:2] + "Z"
    tr.data = np.sum(Z_all, axis=1)
    composite_st.append(tr)
    # For Z mean:
    tr = st[0].copy()
    tr.stats.station = "MEAN"
    tr.stats.channel = st[0].stats.channel[0:2] + "Z"
    tr.data = np.mean(Z_all, axis=1)
    composite_st.append(tr)
    # For Z stdev:
    tr = st[0].copy()
    tr.stats.station = "STDEV"
    tr.stats.channel = st[0].stats.channel[0:2] + "Z"
    tr.data = np.std(Z_all, axis=1)
    composite_st.append(tr)
    # For N stacked:
    tr = st[0].copy()
    tr.stats.station = "STACK"
    tr.stats.channel = st[0].stats.channel[0:2] + "N"
    tr.data = np.sum(N_all, axis=1)
    composite_st.append(tr)
    # For N mean:
    tr = st[0].copy()
    tr.stats.station = "MEAN"
    tr.stats.channel = st[0].stats.channel[0:2] + "N"
    tr.data = np.mean(N_all, axis=1)
    composite_st.append(tr)
    # For N stdev:
    tr = st[0].copy()
    tr.stats.station = "STDEV"
    tr.stats.channel = st[0].stats.channel[0:2] + "N"
    tr.data = np.std(N_all, axis=1)
    composite_st.append(tr)
    # For E stacked:
    tr = st[0].copy()
    tr.stats.station = "STACK"
    tr.stats.channel = st[0].stats.channel[0:2] + "E"
    tr.data = np.sum(E_all, axis=1)
    composite_st.append(tr)
    # For E mean:
    tr = st[0].copy()
    tr.stats.station = "MEAN"
    tr.stats.channel = st[0].stats.channel[0:2] + "E"
    tr.data = np.mean(E_all, axis=1)
    composite_st.append(tr)
    # For E stdev:
    tr = st[0].copy()
    tr.stats.station = "STDEV"
    tr.stats.channel = st[0].stats.channel[0:2] + "E"
    tr.data = np.std(E_all, axis=1)
    composite_st.append(tr)
    del tr
    gc.collect()

    return composite_st


def _create_phase_weighted_stack_st(st, Z_all, N_all, E_all, degree=1):
    """Function to create stacked data st."""

    Z_analytical_signal = hilbert(Z_all, axis=1)
    N_analytical_signal = hilbert(N_all, axis=1)
    E_analytical_signal = hilbert(E_all, axis=1)
    # .  Scipy hilbert transform returns the analytical signal with takes the form:
    # .  s(t) = x(t) + i * y(t) = A(t) * exp(i * phi(t))
    # .  where x(t) is the original signal, y(t) is the Hilbert transform of the signal,
    # .  A(t) is the instantaneous amplitude (envelope) and phi(t) is the instantaneous phase.

    # .  We want to calculate the phase stack, which is given by:
    # .  c(t) = |(1/N) * sum(exp(i * phi_k(t) | ^v
    Z_inst_phase = Z_analytical_signal / np.abs(Z_analytical_signal)
    N_inst_phase = N_analytical_signal / np.abs(N_analytical_signal)
    E_inst_phase = E_analytical_signal / np.abs(E_analytical_signal)

    Z_coherence = np.absolute(np.mean(Z_inst_phase, axis=1))
    N_coherence = np.absolute(np.mean(N_inst_phase, axis=1))
    E_coherence = np.absolute(np.mean(E_inst_phase, axis=1))
    composite_st = obspy.Stream()
    # For Z stacked:
    tr = st[0].copy()
    tr.stats.station = "PW-STACK"
    tr.stats.channel = st[0].stats.channel[0:2] + "Z"
    tr.data = np.mean(Z_all, axis=1) * (Z_coherence**degree)
    print(tr.data.shape)
    composite_st.append(tr)
    # For N stacked:
    tr = st[0].copy()
    tr.stats.station = "PW-STACK"
    tr.stats.channel = st[0].stats.channel[0:2] + "N"
    tr.data = np.mean(N_all, axis=1) * (N_coherence**degree)
    composite_st.append(tr)
    # For E stacked:
    tr = st[0].copy()
    tr.stats.station = "PW-STACK"
    tr.stats.channel = st[0].stats.channel[0:2] + "E"
    tr.data = np.mean(E_all, axis=1) * (E_coherence**degree)
    composite_st.append(tr)
    del tr
    gc.collect()

    return composite_st


class setup_detection:
    """
    Class to create detection object, for running array detection algorithm.

    Parameters
    ----------
    archivedir : str
        Path to data archive. Data archive must be of specific format:
        <archivedir>/YEAR/MONTH/DAY/YYYYMMDDTHHMMSS_*STATION_COMP.*

    outdir : str
        Path to directory to save outputs to.

    stations_fname : str
        Path to csv file containing station/receiver locations. Headers need
        to be of format: Latitude  Longitude  Elevation  Name.

    starttime : obspy UTCDateTime object
        Start time of data window to process for.

    endtime : obspy UTCDateTime object
       End time of data window to process for.

    channels_to_use : list of strs (optional, default = ["??Z"])
        List of channels to use for the processing (e.g. HHZ, ??Z or similar).
        Note: Currently must be in form: ["??Z"] or ["??Z", "??N", "??E"]
        or ["??Z", "??1", "??2"].

    Attributes
    ----------

    skip_existing : float
        If True, then skip exisitng detection time series in outdir. Default is True.

    freqmin : float
        If specified, lower frequency of bandpass filter, in Hz. Default is None.

    freqmax : float
        If specified, upper frequency of bandpass filter, in Hz. Default is None.

    num_freqs : int
        Number of discrete frequencies to use between <freqmin> and <freqmax>
                in analysis. Default is 100. Note: Reducing this value increases
                efficiency linearly, but costs in terms of absolute power. However,
                SNR of power time-series remains approximately constant (to a point).
                Can affect slowness and bazi results, especially within noise, though.

    max_sl : float
        Maximum slowness to analyse for, in s / km. Default is 1.0 s / km.

    win_len_s : float
        The window length for each frequency domain step. Units are seconds.
        Default value is 0.1 s.

    win_step_inc_s : float
        The step increment for each window step. Units are seconds. Note: set this value
        equal to <win_len_s> for no overlap of windows. overlap between windows is given
        by win_len_s - win_step_inc_s. For example, if win_step_inc_s = 3 x <win_len_s>
        then overlap will be 2 x <win_len_s>, etc. Greater overlap gives higher frequency
        resolution, but at computational cost. Default value is 0.1 s.

    remove_autocorr : bool
        If True, then will remove autocorrelations. Default is True.

    norm_pre_stacking : bool
        If True, normallises data before stacking. Similar to performing spectral
        whitening. Default is False

    mad_window_length_s : float
        Length of time-window, in seconds, to calculate background Median Absolute
        Deviation (MAD) for triggering events. Default is 3600 s.

    mad_multiplier : int
        The trigger level above the background MAD level to trigger a phase detection.
        Default = 8.

    min_event_sep_s : float
        Minimum separation between event detections, in seconds. Default = 1 s.

    bazi_tol : float
        The back-azimuth tolerance to associate incoming phases with the same event.
        Various phases have to fulfil the criteria of having the same back-azimuth
        for all phase arrivals, +/- <bazi_tol>. Units are degrees. Defaul is 20 degrees.

    max_phase_sep_s : float
        The maximum time separation between individual phases. Units are seconds. Default
        is 2.5 s.

    filt_phase_assoc_by_max_power : bool
        If True, filters event phase association within a given window by max. power. I.e.
        if True, will only pick highest amplitude phase arrivals on both vertical and
        horizontal components, within the time window <max_phase_sep_s>.

    Methods
    -------

    """

    def __init__(
        self,
        archivedir,
        outdir,
        stations_fname,
        starttime,
        endtime,
        preload_fname=None,
        channels_to_use=["??Z"],
    ):
        """Initiate the class object.

        Parameters
        ----------
        archivedir : str
            Path to data archive. Data archive must be of specific format:
            <archivedir>/YEAR/MONTH/DAY/YYYYMMDDTHHMMSS_*STATION_COMP.*

        outdir : str
            Path to directory to save outputs to.

        stations_fname : str
            Path to csv file containing station/receiver locations. Headers need
            to be of format: Latitude  Longitude  Elevation  Name.

        starttime : obspy UTCDateTime object
            Start time of data window to process for.

        endtime : obspy UTCDateTime object
            End time of data window to process for.

        preload_fname : str
            Path to previously created detection class object. Optional. Default is
            None, which means it doesn't load an existing file.

        channels_to_use : list of strs (optional, default = ["??Z"])
            List of channels to use for the processing (e.g. HHZ, ??Z or similar).

        """
        # Initialise input params:
        self.archivedir = Path(archivedir)
        self.outdir = Path(outdir)
        self.stations_fname = stations_fname
        self.starttime = starttime
        self.endtime = endtime
        self.channels_to_use = channels_to_use

        # Setup outdir:
        os.makedirs(outdir, exist_ok=True)

        # Setup station information:
        self.stations_df = pd.read_csv(self.stations_fname)
        self._setup_array_receiver_coords()

        # And define attributes:
        # For array processing:
        self.skip_existing = True
        self.freqmin = None
        self.freqmax = None
        self.num_freqs = 100
        self.min_sl = 0
        self.max_sl = 1.0
        self.n_sl = 51
        self.min_baz = 0
        self.max_baz = 360
        self.n_baz = 181
        self.win_len_s = 0.1
        self.win_step_inc_s = 0.1  # (Note: Default is to step with no overlap)
        self.remove_autocorr = True
        self.norm_pre_stacking = False
        # self.nproc = 1
        self.out_fnames_array_proc = []
        # For detection:
        self.detector_type = "mad"
        self.mad_window_length_s = 3600
        self.mad_multiplier = 8
        self.selby_filter_half_width_hz = 0.75
        self.selby_filter_step_hz = 0.75
        self.selby_noise_window_s = 30.0
        self.selby_f_threshold = 5.0
        self.selby_f_prominence = None
        self.selby_freqmin = 0.5
        self.selby_freqmax = 6.5
        self.min_event_sep_s = 1.0
        self.bazi_tol = 20.0
        self.max_phase_sep_s = 2.5
        self.min_phase_sep_s = 0
        self.filt_phase_assoc_by_max_power = True
        self.calc_uncertainties = False
        # For location:
        self.receiver_vp = None
        self.receiver_vs = None
        self.array_latlon = None

        # And load existing detection instance, if specified:
        if preload_fname:
            self.load(preload_fname)

    def _detector_suffix(self):
        """Return filename suffix for the active detector."""
        if self.detector_type == "selby":
            return "_selby"
        return ""

    def _build_outfile_name(self, date, hour, channel_curr):
        """Build detector-aware output filenames."""
        suffix = self._detector_suffix()
        return (
            f"detection_t_series_{date.year:02d}{date.month:02d}{date.day:02d}_"
            f"{hour:02d}00_ch{channel_curr[-1]}{suffix}.csv"
        )

    def _get_active_prefilter_band(self):
        """Return the prefilter band used during data loading."""
        if self.detector_type == "selby":
            return self.selby_freqmin, self.selby_freqmax
        return self.freqmin, self.freqmax

    def run_array_proc(self):
        """Function to run core array processing.
        Performed in frequency domain. Involves applying phase (equiv. to time) shift
        for each frequency, over a range of specified slownesses.
        Function inspured by work of D. Bowden (see Bowden et al. (2020))."""

        # Find number of days to run array processing over
        dt_start = self.starttime.date
        dt_end = self.endtime.date
        if (dt_end - dt_start).days == 0:
            # round up to one day at minimum
            ndays = 1
        else:
            ndays = (dt_end - dt_start).days

        query_dates = [dt_start + datetime.timedelta(days=d) for d in range(0, ndays)]
        for date in query_dates:
            # Loop over dates within start/end range:
            # Loop over channels:
            for self.channel_curr in self.channels_to_use:
                logger.info("=" * 60)
                logger.info(
                    f"Processing data for day {date}, channel {self.channel_curr}"
                )
                # And process for individual hours:
                # (to reduce memory usage)
                for hour in range(24):
                    # Loop over every hour in every day..
                    # Make outfile
                    outfile = self._build_outfile_name(date, hour, self.channel_curr)
                    if ((self.outdir / outfile).is_file()) & (self.skip_existing):
                        logger.warning(f"{outfile} exists in {self.outdir}")
                        logger.warning("Move to next hour")
                        self.out_fnames_array_proc.append(f"{self.outdir}/{outfile}")
                        continue
                    if (
                        self.starttime
                        >= obspy.UTCDateTime(
                            year=date.year, month=date.month, day=date.day, hour=hour
                        )
                        + 3600
                    ):
                        continue
                    if self.endtime <= obspy.UTCDateTime(
                        year=date.year, month=date.month, day=date.day, hour=hour
                    ):
                        continue

                    # Create datastores:
                    data_store = {
                        "t": [],
                        "power": [],
                        "score": [],
                        "slowness": [],
                        "back_azi": [],
                        "statistic": [],
                    }

                    # Load data:
                    try:
                        freqmin_override, freqmax_override = (
                            self._get_active_prefilter_band()
                        )
                        st = self._load_data(
                            year=date.year,
                            month=date.month,
                            day=date.day,
                            hour=hour,
                            freqmin_override=freqmin_override,
                            freqmax_override=freqmax_override,
                        )
                    except IndexError:
                        # And skip if no data:
                        logger.exception("Skipping hour as no data")
                        del st
                        gc.collect()
                        continue

                    # And loop over minutes (to save on memory issues):
                    for minute in range(60):
                        # Check whether specified window is greater than a minute in duration:
                        if self.endtime - self.starttime > 60:
                            # Check time within specified run window:
                            if (
                                self.starttime
                                >= obspy.UTCDateTime(
                                    year=date.year,
                                    month=date.month,
                                    day=date.day,
                                    hour=hour,
                                    minute=minute,
                                )
                                + 60
                            ):
                                continue
                            if self.endtime <= obspy.UTCDateTime(
                                year=date.year,
                                month=date.month,
                                day=date.day,
                                hour=hour,
                                minute=minute,
                            ):
                                continue
                        elif self.starttime.minute != minute:
                            continue

                        # Trim data:
                        st_trimmed = st.copy()
                        if self.win_len_s > self.win_step_inc_s:
                            self.win_pad_s = self.win_len_s
                        else:
                            self.win_pad_s = 0.0
                        if self.endtime - self.starttime > 60:
                            st_trimmed.trim(
                                starttime=obspy.UTCDateTime(
                                    year=date.year,
                                    month=date.month,
                                    day=date.day,
                                    hour=hour,
                                    minute=minute,
                                ),
                                endtime=obspy.UTCDateTime(
                                    year=date.year,
                                    month=date.month,
                                    day=date.day,
                                    hour=hour,
                                    minute=minute,
                                )
                                + 60
                                + self.win_pad_s,
                            )
                        else:
                            st_trimmed.trim(
                                starttime=self.starttime,
                                endtime=self.endtime + self.win_pad_s,
                            )
                        time_this_minute_st = st_trimmed[0].stats.starttime
                        # Run array processing:
                        if self.detector_type == "selby":
                            score_cube = self._selby_beamforming(st_trimmed)
                            statistic = "selby_f"
                        else:
                            score_cube = self._beamforming(st_trimmed)
                            statistic = "beam_power"
                        del st_trimmed
                        gc.collect()

                        # Calculate time-series outputs (for detection) from data:
                        t_series, powers, slownesses, back_azis = (
                            self._find_time_series_from_score_cube(score_cube)
                        )
                        # And append to data out:
                        t_series_out = []
                        for t_serie in t_series:
                            t_series_out.append(str(time_this_minute_st + t_serie))

                        data_store["t"].extend(t_series_out)
                        data_store["power"].extend(powers)
                        data_store["score"].extend(powers)
                        data_store["slowness"].extend(slownesses)
                        data_store["back_azi"].extend(back_azis)
                        data_store["statistic"].extend([statistic] * len(powers))

                        # And clear memory:
                        del score_cube, t_series, powers, slownesses, back_azis
                        gc.collect()

                    # And save data out:
                    out_fname = os.path.join(self.outdir, outfile)
                    # make DataFrame "just-in-time" as it is more efficient this way
                    store_df = pd.DataFrame(data_store)
                    store_df.to_csv(out_fname, index=False)

                    # And append fname to history:
                    self.out_fnames_array_proc.append(out_fname)

        return None

    def _setup_array_receiver_coords(self):
        """Function to setup station receiver coords in correct format for
        array processing."""
        ref_lat = np.mean(self.stations_df["Latitude"])
        ref_lon = np.mean(self.stations_df["Longitude"])
        # Calculate receiver locations in km grid format:
        xs = []
        ys = []
        for index, row in self.stations_df.iterrows():
            lon2 = row["Longitude"]
            lat2 = row["Latitude"]
            # Calc. inter-station distance:
            r, a, b = obspy.geodetics.base.gps2dist_azimuth(
                ref_lat, ref_lon, lat2, lon2
            )
            xs.append(r * np.sin(np.deg2rad(a)) / 1000)
            ys.append(r * np.cos(np.deg2rad(a)) / 1000)
        self.stations_df["x (km)"] = xs
        self.stations_df["y (km)"] = ys
        self.stations_df["z (km)"] = self.stations_df["Elevation"].values / 1000.0
        self.stations_df["ref_lat"] = np.ones(len(xs)) * ref_lat
        self.stations_df["ref_lon"] = np.ones(len(xs)) * ref_lon

        # And calculate receiver locations in terms of array centre:
        array_centre = np.array(
            [
                np.mean(self.stations_df["x (km)"]),
                np.mean(self.stations_df["y (km)"]),
                np.mean(self.stations_df["z (km)"]),
            ]
        )
        self.stations_df["x_array_coords_km"] = (
            self.stations_df["x (km)"].values - array_centre[0]
        )
        self.stations_df["y_array_coords_km"] = (
            self.stations_df["y (km)"].values - array_centre[1]
        )
        self.stations_df["z_array_coords_km"] = (
            self.stations_df["z (km)"].values - array_centre[2]
        )
        # And in polar coords from N:
        (
            self.stations_df["r_array_coords_km"],
            self.stations_df["theta_array_coords_deg"],
        ) = xy_to_rtheta(
            self.stations_df["x_array_coords_km"], self.stations_df["y_array_coords_km"]
        )

    def find_min_max_array_sensitivity(self, vel_assumed=3.0):
        """Function to find array min and max sensitivities.
        Parameters
        ----------
        vel_assumed : float (default = 3.0)
            Approx. velocity of seismic waves incident at the array, in km/s.
        """
        # And find min and max distance (and hence frequency) for the array:
        inter_station_dists = []
        # Loop over first set of stations:
        for index, row in self.stations_df.iterrows():
            lon1 = row["Longitude"]
            lat1 = row["Latitude"]
            # Loop over stations again:
            for index, row in self.stations_df.iterrows():
                lon2 = row["Longitude"]
                lat2 = row["Latitude"]
                # Calc. inter-station distance:
                r, a, b = obspy.geodetics.base.gps2dist_azimuth(lat1, lon1, lat2, lon2)
                inter_station_dists.append(np.abs(r) / 1000)
        inter_station_dists = np.array(inter_station_dists)
        inter_station_dists = inter_station_dists[inter_station_dists != 0]
        print("=" * 60)
        print("Min. inter-station distance:", np.min(inter_station_dists), "km")
        print(
            "Therefore, optimal sensitive higher freq.:",
            vel_assumed / np.min(inter_station_dists),
            "Hz",
        )
        print("Max. inter-station distance:", np.max(inter_station_dists), "km")
        print(
            "Therefore, optimal sensitive lower freq.:",
            vel_assumed / np.max(inter_station_dists),
            "Hz",
        )
        print("=" * 60)

    def _load_data(
        self, year, month, day, hour=None, norm=True, freqmin_override=None, freqmax_override=None
    ):
        """
        Function to load data. If no hour is specified the whole day will be read in.
        Otherwise the hour of data will be loaded.

        Parameters
        ----------

        year : int
            year to load data for (yyyy)
        month : int
            month to load data for. Leading 0's will be added.
        day : int
            day to load data for. Leading 0's will be added.
        hour : int, Optional
            hour to load data for. Leading 0's will be added.

        Returns:
        ----------
        data : obspy.Stream
            Waveform data for requested date and time.
        """
        # Load in data:
        mseed_dir = Path(
            self.archivedir, str(year), str(month).zfill(2), str(day).zfill(2)
        )
        # print(mseed_dir)
        st = obspy.Stream()
        for index, row in self.stations_df.iterrows():
            # [J Asplet - think about replacing station DataFrame with StatonXML object]
            station = row["Name"]
            for channel in self.channels_to_use:
                if hour is None:
                    timestamp = f"{year:02d}{month:02d}{day:02d}T*"
                else:
                    timestamp = f"{year:02d}{month:02d}{day:02d}T{hour:02d}0000"
                try:
                    st_tmp = obspy.read(
                        f"{mseed_dir}/{timestamp}_{station}_{channel}.mseed"
                    )
                    for tr in st_tmp:
                        st.append(tr)
                except:
                    logger.exception(
                        f"No data for {station}, channel = {channel}, timestamp {timestamp}. Skipping this data."
                    )
                    continue
        # Merge data:
        st.detrend("demean")
        st.detrend("linear")
        st.merge(method=1, fill_value=0.0)
        # And apply filter:
        freqmin_to_use = self.freqmin if freqmin_override is None else freqmin_override
        freqmax_to_use = self.freqmax if freqmax_override is None else freqmax_override
        if freqmin_to_use and freqmax_to_use:
            st.filter("bandpass", freqmin=freqmin_to_use, freqmax=freqmax_to_use)
        # And trim data, if some lies outside start and end time of beamforming period:
        if self.starttime > st[0].stats.starttime:
            st.trim(starttime=self.starttime)
        if self.endtime < st[0].stats.endtime:
            st.trim(endtime=self.endtime)

        return st

    def _convert_st_to_np_data(self, st):
        """Function to convert data to numpy format for processing."""
        self.n_win = int(
            ((st[0].stats.endtime - self.win_pad_s) - st[0].stats.starttime)
            / self.win_step_inc_s
        )  # (Note: endtime - self.win_pad_s as pass extra padding via trimmed st)
        self.fs = st[0].stats.sampling_rate
        self.n_t_samp = int(self.win_len_s * self.fs)  # num samples in time
        station_labels = self.stations_df["Name"].values
        self.n_stations = len(station_labels)  # num stations
        data = np.zeros((self.n_win, self.n_stations, self.n_t_samp))
        for i in range(self.n_win):
            for j in range(self.n_stations):
                station = station_labels[j]
                win_start_idx = i * int(self.win_step_inc_s * self.fs)
                win_end_idx = (i * int(self.win_step_inc_s * self.fs)) + self.n_t_samp
                try:
                    if win_end_idx < len(
                        st.select(station=station, channel=self.channel_curr)[0].data
                    ):
                        data[i, j, :] = st.select(
                            station=station, channel=self.channel_curr
                        )[0].data[win_start_idx:win_end_idx]
                    else:
                        # Zero pad data (as insufficient data passed for final window) and print warning:
                        data[i, j, :] = 0.0
                        logger.warning(
                            f"Warning: Zero-padding as not enough data to fill window overlap ( for win_len_s = {self.win_len_s}, and win_step_inc_s = {self.win_step_inc_s})"
                        )
                except IndexError:
                    # Deal with if a particular station has no data for given window:
                    data[i, j, :] = 0.0
        return data

    def _convert_st_to_station_matrix(self, st):
        """Convert a trimmed stream to a station x sample matrix."""
        station_labels = self.stations_df["Name"].values
        data_len = len(st[0].data)
        data = np.zeros((len(station_labels), data_len))
        for idx, station in enumerate(station_labels):
            traces = st.select(station=station, channel=self.channel_curr)
            if len(traces) > 0:
                tr = traces[0]
                n_copy = min(len(tr.data), data_len)
                data[idx, :n_copy] = tr.data[:n_copy]
        return data

    def _stack_results(self, Pfreq_all):
        """Function to perform stacking of the results."""
        Psum_all = np.zeros(
            (Pfreq_all.shape[0], Pfreq_all.shape[2], Pfreq_all.shape[3]), dtype=complex
        )
        # Loop over time windows:
        for i in range(Pfreq_all.shape[0]):
            if self.norm_pre_stacking:
                Pfreq_norm_curr = Pfreq_all[i, :, :, :] / np.sum(
                    np.abs(Pfreq_all[i, :, :, :]), axis=0
                )
                Psum_all[i, :, :] = np.sum(Pfreq_norm_curr, axis=0)
            else:
                Psum_all[i, :, :] = np.sum(Pfreq_all[i, :, :, :], axis=0)
        return Psum_all

    def _find_time_series(self, Psum_all):
        """Function to calculate beamforming time-series outputs, given
        a raw beamforming result.
        Note that the time-series timestamps are in the middle of the time-
        wimdows.
        Psum_all - Sum of Pxx for each time window, and for all slownesses.
                    Shape is n_win x slowness WE x slowness SN
        Returns time-series of coherency (power), slowness and back-azimuth.
        """
        return self._find_time_series_from_score_cube(Psum_all)

    def _find_time_series_from_score_cube(self, score_cube):
        """Extract detector score, slowness and back-azimuth from a score cube."""
        # Calcualte ux, uy:
        ur = np.linspace(self.min_sl, self.max_sl, score_cube.shape[1])
        utheta = np.linspace(
            self.min_baz,
            self.max_baz - (self.max_baz / score_cube.shape[2]),
            score_cube.shape[2],
        )
        # Create time-series:
        n_win_curr = score_cube.shape[0]
        t_series = np.arange(
            self.win_step_inc_s / 2,
            (n_win_curr * self.win_step_inc_s) + (self.win_step_inc_s / 2),
            self.win_step_inc_s,
        )
        if len(t_series) > n_win_curr:
            t_series = t_series[0:n_win_curr]
        # And find power, slowness and back-azimuth:
        powers = np.zeros(n_win_curr)
        slownesses = np.zeros(n_win_curr)
        back_azis = np.zeros(n_win_curr)
        # Loop over windows in time:
        for i in range(n_win_curr):
            powers[i] = np.max(np.abs(score_cube[i, :, :]))
            # Calculate slowness:
            r_idx = np.where(score_cube[i, :, :] == score_cube[i, :, :].max())[0][0]
            theta_idx = np.where(score_cube[i, :, :] == score_cube[i, :, :].max())[1][0]
            slownesses[i] = ur[r_idx]
            # And calculate back-azimuth:
            back_azis[i] = utheta[theta_idx]

        return t_series, powers, slownesses, back_azis

    def _selby_beamforming(self, st_trimmed):
        """Calculate a Selby-inspired detector score cube for a trimmed stream."""
        station_matrix = self._convert_st_to_station_matrix(st_trimmed)
        xx = self.stations_df["x_array_coords_km"].values
        yy = self.stations_df["y_array_coords_km"].values
        score_cube, _ = selby.compute_selby_score_cube(
            station_matrix,
            st_trimmed[0].stats.sampling_rate,
            xx,
            yy,
            self.min_sl,
            self.max_sl,
            self.n_sl,
            self.min_baz,
            self.max_baz,
            self.n_baz,
            self.win_len_s,
            self.win_step_inc_s,
            self.selby_freqmin,
            self.selby_freqmax,
            self.selby_filter_step_hz,
            self.selby_filter_half_width_hz,
            self.selby_noise_window_s,
        )
        return score_cube

    def _beamforming(self, st_trimmed, verbosity=0):
        """Function to perform beamforming, given a stream of data for a specific
        time-window. Function is primarily called by run_array_proc().
        Returns <Psum_all> (stacked 2D power-slowness space data)."""
        # Run heavy array processing algorithm:
        # Specify various variables needed:
        # Make a linear spacing of frequencies. One might use periods, logspacing, etc.:
        # (Note: linspace much less noisy than logspace)
        target_freqs = np.linspace(
            self.freqmin, self.freqmax, self.num_freqs
        )  # np.logspace(self.freqmin,self.freqmax,self.num_freqs)
        data = self._convert_st_to_np_data(st_trimmed)
        # Station locations:
        xx = self.stations_df["x_array_coords_km"].values
        yy = self.stations_df["y_array_coords_km"].values
        # And run:
        if verbosity > 1:
            logger.info("Performing run for", data.shape[0], "windows")
            tic = time.time()
        Pfreq_all = _fast_freq_domain_array_proc(
            data,
            self.min_sl,
            self.max_sl,
            self.n_sl,
            self.min_baz,
            self.max_baz,
            self.n_baz,
            self.fs,
            target_freqs,
            xx,
            yy,
            self.n_stations,
            self.n_t_samp,
            self.remove_autocorr,
        )
        if verbosity > 1:
            toc = time.time()
            logger.info(f"runtime for _beamforming is {toc-tic}")
        # And tidy:
        del data
        gc.collect()

        # And remove any data where stations don't exist:
        Pfreq_all = np.nan_to_num(Pfreq_all, copy=False, nan=0.0)

        # Stack (and normalise) data:
        Psum_all = self._stack_results(Pfreq_all)
        del Pfreq_all
        gc.collect()

        return Psum_all

    def _calculate_mad(self, x, scale=1.4826):
        """
        Calculates the Median Absolute Deviation (MAD) of the input array x.
        Outputs an array of scaled mean absolute deviation values for the
        input array, x, scaled to provide an estimation of the standard
        deviation of the distribution.
        """
        # Calculate median and mad values:
        mad = np.median(np.abs(x - np.median(x)))
        return scale * mad

    def plot_polar_slowness_space(
        self, beam_power, event_phase_arr_time, component, log=False
    ):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="polar")
        rad = np.linspace(self.min_sl, self.max_sl, beam_power.shape[0])
        azm = np.linspace(
            np.radians(self.min_baz), np.radians(self.max_baz), beam_power.shape[1]
        )
        th, r = np.meshgrid(azm, rad)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        if log:
            im = ax.pcolormesh(th, r, np.log(beam_power), cmap="magma")
        else:
            im = ax.pcolormesh(th, r, beam_power, cmap="magma")

        plt.colorbar(im)
        plt.grid()
        event_date_stamp = f"{event_phase_arr_time.year:04d}{event_phase_arr_time.month:02d}{event_phase_arr_time.day:02d}"
        event_time_stamp = f"{event_phase_arr_time.hour:02d}{event_phase_arr_time.minute:02d}{event_phase_arr_time.second:02d}"
        vesp_figpath = Path(self.outdir, "plots", "vespagrams")
        vesp_figpath.mkdir(
            parents=True, exist_ok=True
        )  # makes plots/vespagrams if it doesnt exist
        fig.savefig(
            f"{vesp_figpath}/Detected_event_{event_date_stamp}_{event_time_stamp}_slow_spac_{component}.png",
            dpi=600,
        )
        plt.close()

    def _calc_uncertainties(
        self, events_df, t_series_df_Z, t_series_df_hor, verbosity=0
    ):
        """Function to calculate uncertainties for phase-associated event detections.
        Calculates uncertainty in t1, t2, slow1, slow2 and bazi1 and bazi2,
        assuming Gaussian uncertainties. Uncertainties are defined as the full-width half
        maximum (used due to more efficient optimisation than Gaussian fitting, but potentially
        over-estimates error)."""
        # Do preliminary prep. once:
        # Define temparory datastore:
        uncertainties_df = pd.DataFrame(
            {
                "t1_err": [],
                "t2_err": [],
                "slow1_err": [],
                "slow2_err": [],
                "bazi1_err": [],
                "bazi2_err": [],
            }
        )
        # Find max. timeshift (for determining beamforming window):
        max_t_shift = self.max_sl * (
            np.max(np.abs((self.stations_df["x_array_coords_km"].values)))
            + np.max(np.abs((self.stations_df["x_array_coords_km"].values)))
        )  # (effectively d/v)
        n_wins_for_max_t_shift = int(
            np.ceil(max_t_shift / self.win_step_inc_s)
        )  # + 1 #(+1 just to ensure that window is definitely wide enough)

        # And loop over detected events, calculating uncertainty:
        count = 0
        for index, row in events_df.iterrows():
            if count % 10 == 0:
                if verbosity > 0:
                    logger.info(
                        "Calculating uncertainty for event",
                        count + 1,
                        "/",
                        len(events_df),
                    )
            # Load in data (if needed):
            # (done like this to avoid unnneccessary read ins, improving eff.)
            event_phase_arr_time = obspy.UTCDateTime(row["t1"])
            if count == 0:
                st = self._load_data(
                    event_phase_arr_time.year,
                    event_phase_arr_time.month,
                    event_phase_arr_time.day,
                    hour=event_phase_arr_time.hour,
                )

            # Find uncertainties:
            # ------- For vertical -------:
            # Time uncertainty:
            # Find FWHM for t1 pick:
            # (only use ascending currently (assume symetric pdf))
            t1_pick_idx = t_series_df_Z.index[t_series_df_Z["t"] == row["t1"]][0]
            Pxx_curr = t_series_df_Z.iloc[t1_pick_idx]["power"]
            idx_diff = 0
            while Pxx_curr > t_series_df_Z.iloc[t1_pick_idx]["power"] / 2.0:
                idx_diff += 1
                Pxx_curr = t_series_df_Z.iloc[t1_pick_idx + idx_diff]["power"]
            t1_err = obspy.UTCDateTime(
                t_series_df_Z.iloc[t1_pick_idx + idx_diff]["t"]
            ) - obspy.UTCDateTime(t_series_df_Z.iloc[t1_pick_idx]["t"])

            # Spatial uncertainty:
            # (slowness, bazi)
            # Perform beamforming again around event, to estimate bazi and slowness errs:
            # Get data:
            event_phase_arr_time = obspy.UTCDateTime(row["t1"])
            # Reload data if needed:
            if (
                st[0].stats.starttime > event_phase_arr_time
                or st[0].stats.endtime < event_phase_arr_time
            ):
                st = self._load_data(
                    event_phase_arr_time.year,
                    event_phase_arr_time.month,
                    event_phase_arr_time.day,
                    hour=event_phase_arr_time.hour,
                )
            st_trimmed = st.copy()
            st_trimmed.trim(
                starttime=event_phase_arr_time
                - ((n_wins_for_max_t_shift + 0.5) * self.win_len_s),
                endtime=event_phase_arr_time
                + ((n_wins_for_max_t_shift + 0.5) * self.win_len_s),
            )  # (Note: 0.5 as windows centred)
            # Run array processing:
            # (to get power in polar slowness space)
            # (Note that need to run for a number of windows, to allow for adequate shifting of data)
            self.channel_curr = self.channels_to_use[0]  # Do for vertical first
            Psum_all = self._beamforming(st_trimmed, verbosity=verbosity)
            del st_trimmed
            gc.collect()
            # Find highest power slowness space for event:
            t_series, powers, slownesses, back_azis = self._find_time_series(Psum_all)
            max_idx = np.argmax(powers)
            Psum_opt = np.abs(Psum_all[max_idx, :, :])
            # Find FWHM for slowness and bazi:
            slow_idx_peak = np.where(Psum_opt == np.max(Psum_opt))[0][0]
            bazi_idx_peak = np.where(Psum_opt == np.max(Psum_opt))[1][0]
            # Slowness:
            # (go radially outwards for slowness, assumes symetric or sharper gradient inwards)
            Pxx_curr = Psum_opt[slow_idx_peak, bazi_idx_peak]
            idx_diff = 0
            while Pxx_curr > Psum_opt[slow_idx_peak, bazi_idx_peak] / 2.0:
                idx_diff += 1
                if slow_idx_peak + idx_diff < Psum_opt.shape[0]:
                    Pxx_curr = Psum_opt[slow_idx_peak + idx_diff, bazi_idx_peak]
                else:
                    Pxx_curr = 0  # Force exit if reach slowness limits
            dslow = self.max_sl / Psum_opt.shape[0]  # Assumes linear slowness space
            slow1_err = idx_diff * dslow
            # Back-azimuth:
            # (go clockwise, assuming symetric)
            Pxx_curr = Psum_opt[slow_idx_peak, bazi_idx_peak]
            idx_diff = 0
            while Pxx_curr > Psum_opt[slow_idx_peak, bazi_idx_peak] / 2.0:
                idx_diff += 1
                if bazi_idx_peak + idx_diff < Psum_opt.shape[1]:
                    Pxx_curr = Psum_opt[slow_idx_peak, bazi_idx_peak + idx_diff]
                else:
                    try:
                        Pxx_curr = Psum_opt[
                            slow_idx_peak, bazi_idx_peak + idx_diff - Psum_opt.shape[1]
                        ]  # (loop beyond 360 degrees)
                    except IndexError:
                        # Deal with 360 degree error:
                        Pxx_curr = 0
                        idx_diff = Psum_opt.shape[1]
            dbazi = 360 / Psum_opt.shape[1]

            bazi1_err = idx_diff * dbazi
            # ------- End vertical -------

            # Plot slowness space that used for uncertainty, if specified:
            if verbosity >= 1:
                self.plot_polar_slowness_space(
                    Psum_opt, event_phase_arr_time, component="vert"
                )
            # ------- For horizontal -------:
            # And find FWHM for t2 pick:
            # (only use ascending currently (assume symetric pdf))
            t2_pick_idx = t_series_df_hor.index[t_series_df_hor["t"] == row["t2"]][0]
            Pxx_curr = t_series_df_hor.iloc[t2_pick_idx]["power"]
            idx_diff = 0
            while Pxx_curr > t_series_df_hor.iloc[t2_pick_idx]["power"] / 2.0:
                idx_diff += 1
                Pxx_curr = t_series_df_hor.iloc[t2_pick_idx + idx_diff]["power"]
            t2_err = obspy.UTCDateTime(
                t_series_df_hor.iloc[t2_pick_idx + idx_diff]["t"]
            ) - obspy.UTCDateTime(t_series_df_hor.iloc[t2_pick_idx]["t"])

            # Spatial uncertainty:
            # (slowness, bazi)
            # Perform beamforming again around event, to estimate bazi and slowness errs:
            # Get data:
            event_phase_arr_time = obspy.UTCDateTime(row["t2"])
            # Reload data if needed:
            if (
                st[0].stats.starttime > event_phase_arr_time
                or st[0].stats.endtime < event_phase_arr_time
            ):
                st = self._load_data(
                    event_phase_arr_time.year,
                    event_phase_arr_time.month,
                    event_phase_arr_time.day,
                    hour=event_phase_arr_time.hour,
                )
            st_trimmed = st.copy()
            st_trimmed.trim(
                starttime=event_phase_arr_time
                - ((n_wins_for_max_t_shift + 0.5) * self.win_len_s),
                endtime=event_phase_arr_time
                + ((n_wins_for_max_t_shift + 0.5) * self.win_len_s),
            )  # (Note: 0.5 as windows centred)
            # Run array processing:
            # (to get power in polar slowness space)
            # (Note that need to run for a number of windows, to allow for adequate shifting of data)
            self.channel_curr = self.channels_to_use[1]  # Do for vertical first
            Psum_all_N = self._beamforming(st_trimmed, verbosity=verbosity)
            self.channel_curr = self.channels_to_use[2]  # Do for vertical first
            Psum_all_E = self._beamforming(st_trimmed, verbosity=verbosity)
            del st_trimmed
            gc.collect()
            Psum_all_NE = Psum_all_N + Psum_all_E
            # Find highest power slowness space for event:
            t_series, powers, slownesses, back_azis = self._find_time_series(
                Psum_all_NE
            )
            max_idx = np.argmax(powers)
            Psum_opt = np.abs(Psum_all_NE[max_idx, :, :])
            # Find FWHM for slowness and bazi:
            slow_idx_peak = np.where(Psum_opt == np.max(Psum_opt))[0][0]
            bazi_idx_peak = np.where(Psum_opt == np.max(Psum_opt))[1][0]
            # Slowness:
            # (go radially outwards for slowness, assumes symetric or sharper gradient inwards)
            Pxx_curr = Psum_opt[slow_idx_peak, bazi_idx_peak]
            idx_diff = 0
            while Pxx_curr > Psum_opt[slow_idx_peak, bazi_idx_peak] / 2.0:
                idx_diff += 1
                if slow_idx_peak + idx_diff < Psum_opt.shape[0]:
                    Pxx_curr = Psum_opt[slow_idx_peak + idx_diff, bazi_idx_peak]
                else:
                    Pxx_curr = 0  # Force exit if reach slowness limits
            dslow = self.max_sl / Psum_opt.shape[0]  # Assumes linear slowness space
            slow2_err = idx_diff * dslow
            # Back-azimuth:
            # (go clockwise, assuming symetric)
            Pxx_curr = Psum_opt[slow_idx_peak, bazi_idx_peak]
            idx_diff = 0
            while Pxx_curr > Psum_opt[slow_idx_peak, bazi_idx_peak] / 2.0:
                idx_diff += 1
                if bazi_idx_peak + idx_diff < Psum_opt.shape[1]:
                    Pxx_curr = Psum_opt[slow_idx_peak, bazi_idx_peak + idx_diff]
                else:
                    try:
                        Pxx_curr = Psum_opt[
                            slow_idx_peak, bazi_idx_peak + idx_diff - Psum_opt.shape[1]
                        ]  # (loop beyond 360 degrees)
                    except IndexError:
                        # Deal with 360 degree error:
                        Pxx_curr = 0
                        idx_diff = Psum_opt.shape[1]
            dbazi = 360.0 / Psum_opt.shape[1]
            bazi2_err = idx_diff * dbazi
            # ------- End horizontal -------

            # Plot slowness space that used for uncertainty, if specified:
            if verbosity >= 1:
                self.plot_polar_slowness_space(
                    Psum_opt, event_phase_arr_time, component="horz"
                )
            # And append data to overall uncertainties df:
            uncertainties_df_curr = pd.DataFrame(
                {
                    "t1_err": [t1_err],
                    "t2_err": [t2_err],
                    "slow1_err": [slow1_err],
                    "slow2_err": [slow2_err],
                    "bazi1_err": [bazi1_err],
                    "bazi2_err": [bazi2_err],
                }
            )
            uncertainties_df = pd.concat(
                [uncertainties_df, uncertainties_df_curr], ignore_index=True
            )

            # And update count:
            count += 1

        # And add uncertainties to events_df:
        events_df = events_df.reset_index(drop=True)
        uncertainties_df = uncertainties_df.reset_index(drop=True)
        events_df = pd.concat([events_df, uncertainties_df], axis=1)

        return events_df

    def detect_events(self, verbosity=0, fnames=None):
        """Function to detect events, based on the power time-series generated
        by run_array_proc(). Note: Currently, only Median Absolute Deviation
        triggering is implemented.
        Key attributes used are:
        - mad_window_length_s
        - mad_multiplier
        - min_event_sep_s
        """
        # print("Note: <mad_window_length_s> not yet implemented.")
        # Create datastore:
        events_df_all = pd.DataFrame()
        # Loop over array proc outdir data:
        if fnames is None:
            detector_suffix = self._detector_suffix()
            fnames = glob.glob(
                os.path.join(
                    self.outdir, f"detection_t_series_*_chZ{detector_suffix}.csv"
                )
            )
        for fname in sorted(fnames):
            basename = os.path.basename(fname)
            channel_token = "_chZ"
            if channel_token not in basename:
                continue
            f_uid = basename.split(channel_token)[0].replace("detection_t_series_", "")
            # Check if in list to process:
            if fname in self.out_fnames_array_proc:
                # And load in data:
                # Read in vertical data:
                t_series_df_Z = pd.read_csv(
                    fname, converters={"t": lambda x: obspy.UTCDateTime(x)}
                )
                t_series_df_Z = _ensure_score_columns(t_series_df_Z)
                # And read in horizontals:
                detector_suffix = self._detector_suffix()
                try:
                    fname_N = os.path.join(
                        self.outdir,
                        "".join(
                            ("detection_t_series_", f_uid, "_chN", detector_suffix, ".csv")
                        ),
                    )
                    t_series_df_N = pd.read_csv(
                        fname_N, converters={"t": lambda x: obspy.UTCDateTime(x)}
                    )
                    t_series_df_N = _ensure_score_columns(t_series_df_N)
                except FileNotFoundError:
                    fname_N = os.path.join(
                        self.outdir,
                        "".join(
                            ("detection_t_series_", f_uid, "_ch1", detector_suffix, ".csv")
                        ),
                    )
                    t_series_df_N = pd.read_csv(
                        fname_N, converters={"t": lambda x: obspy.UTCDateTime(x)}
                    )
                    t_series_df_N = _ensure_score_columns(t_series_df_N)
                try:
                    fname_E = os.path.join(
                        self.outdir,
                        "".join(
                            ("detection_t_series_", f_uid, "_chE", detector_suffix, ".csv")
                        ),
                    )
                    t_series_df_E = pd.read_csv(
                        fname_E, converters={"t": lambda x: obspy.UTCDateTime(x)}
                    )
                    t_series_df_E = _ensure_score_columns(t_series_df_E)
                except FileNotFoundError:
                    fname_E = os.path.join(
                        self.outdir,
                        "".join(
                            ("detection_t_series_", f_uid, "_ch2", detector_suffix, ".csv")
                        ),
                    )
                    t_series_df_E = pd.read_csv(
                        fname_E, converters={"t": lambda x: obspy.UTCDateTime(x)}
                    )
                    t_series_df_E = _ensure_score_columns(t_series_df_E)
            else:
                logger.warning(f"fname {fname} not in fname_array_proc list")
                logger.warning(
                    f"fname_array_proc first entry looks like this {self.out_fnames_array_proc[0]}"
                )
                continue  # Skip file, as not previously been processed.
            # And check to see that t-series exists within file:
            if len(t_series_df_Z) == 0:
                continue
            if len(t_series_df_N) == 0:
                continue
            if len(t_series_df_E) == 0:
                continue

            # Check if all inputs are same length, and if not, skip file:
            if not len(t_series_df_Z) == len(t_series_df_N) == len(t_series_df_E):
                logger.warning(
                    "Warning: Files with f uid",
                    f_uid,
                    "are not of equal length. Therefore using shortest length (will miss some data).",
                )
                logger.warning(
                    "( Lengths are",
                    len(t_series_df_Z),
                    len(t_series_df_N),
                    len(t_series_df_E),
                    ")",
                )
                min_len = np.min(
                    np.array(
                        [len(t_series_df_Z), len(t_series_df_N), len(t_series_df_E)]
                    )
                )
                t_series_df_Z = t_series_df_Z.iloc[:min_len]
                t_series_df_N = t_series_df_N.iloc[:min_len]
                t_series_df_E = t_series_df_E.iloc[:min_len]

            # Combine horizontals:
            # (Using RMS of N and E signals for slowness and average for BAZI)
            t_series_df_hor = t_series_df_N.copy()
            t_series_df_hor["power"] = np.sqrt(
                t_series_df_N["power"].values ** 2 + t_series_df_E["power"].values ** 2
            )
            t_series_df_hor["score"] = np.sqrt(
                t_series_df_N["score"].values ** 2 + t_series_df_E["score"].values ** 2
            )
            NE_Pxx_max = np.max(
                np.concatenate(
                    (t_series_df_N["power"].values, t_series_df_E["power"].values)
                )
            )
            N_weighting = t_series_df_N["power"].values / NE_Pxx_max
            E_weighting = t_series_df_E["power"].values / NE_Pxx_max
            t_series_df_hor["slowness"] = np.sqrt(
                np.average(
                    np.vstack(
                        (t_series_df_N["slowness"] ** 2, t_series_df_E["slowness"] ** 2)
                    ),
                    axis=0,
                    weights=np.vstack((N_weighting, E_weighting)),
                )
            )  # Weighted mean (weighted by power)
            t_series_df_hor["back_azi"] = np.average(
                np.vstack((t_series_df_N["back_azi"], t_series_df_E["back_azi"])),
                axis=0,
                weights=np.vstack((N_weighting, E_weighting)),
            )  # Weighted mean (weighted by power)
            # print("(Weighted horizontal slowness and back-azi using power)")
            del N_weighting, E_weighting, t_series_df_N, t_series_df_E
            gc.collect()

            # Get phase picks:
            if len(t_series_df_Z) < 2:
                continue
            min_pick_dist = int(
                max(
                    1,
                    self.min_event_sep_s
                    / (
                        obspy.UTCDateTime(t_series_df_Z["t"][1])
                        - obspy.UTCDateTime(t_series_df_Z["t"][0])
                    ),
                )
            )
            if self.detector_type == "selby":
                selby_prominence = (
                    self.selby_f_threshold
                    if self.selby_f_prominence is None
                    else self.selby_f_prominence
                )
                peaks_Z, _ = find_peaks(
                    t_series_df_Z["score"].values,
                    height=self.selby_f_threshold,
                    distance=min_pick_dist,
                    prominence=selby_prominence,
                )
                peaks_hor, _ = find_peaks(
                    t_series_df_hor["score"].values,
                    height=self.selby_f_threshold,
                    distance=min_pick_dist,
                    prominence=selby_prominence,
                )
            else:
                mad_pick_threshold_Z = moving_window_mad(
                    t_series_df_Z["power"].values,
                    self.mad_window_length_s,
                    self.mad_multiplier,
                )

                mad_pick_threshold_hor = moving_window_mad(
                    t_series_df_hor["power"].values,
                    self.mad_window_length_s,
                    self.mad_multiplier,
                )
                peaks_Z, _ = find_peaks(
                    t_series_df_Z["power"].values,
                    height=mad_pick_threshold_Z,
                    distance=min_pick_dist,
                    prominence=mad_pick_threshold_Z,
                )
                peaks_hor, _ = find_peaks(
                    t_series_df_hor["power"].values,
                    height=mad_pick_threshold_hor,
                    distance=min_pick_dist,
                    prominence=mad_pick_threshold_hor,
                )
            print(
                f"Found {len(peaks_Z)} P-phase picks and {len(peaks_hor)} S-phase picks for file with uid {f_uid}"
            )
            # Phase associate by BAZI threshold and max. power:
            events_df = _phase_associator(
                t_series_df_Z,
                t_series_df_hor,
                peaks_Z,
                peaks_hor,
                self.bazi_tol,
                self.filt_phase_assoc_by_max_power,
                self.max_phase_sep_s,
                self.min_phase_sep_s,
                self.min_event_sep_s,
                verbosity=verbosity,
            )

            # Find uncertainties (in time, bazi, slowness):
            if self.calc_uncertainties:
                if self.detector_type == "selby":
                    raise NotImplementedError(
                        "Selby-inspired detections do not yet support uncertainty "
                        "estimation via the legacy beam-power workflow."
                    )
                events_df = self._calc_uncertainties(
                    events_df, t_series_df_Z, t_series_df_hor, verbosity=verbosity
                )
            # Plot detected, phase-associated picks:
            if verbosity > 1:
                # print("="*40)
                logger.info("Event phase associations:")
                # print(events_df)
                # print("="*40)
                time_after_startZ = np.array(t_series_df_Z["t"]) - t_series_df_Z["t"][0]
                time_after_startH = (
                    np.array(t_series_df_hor["t"]) - t_series_df_hor["t"][0]
                )

                fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(9, 6))
                # Plot power:
                ax[0].plot(
                    time_after_startZ, t_series_df_Z["power"], label="Vertical power"
                )
                ax[0].plot(
                    time_after_startH,
                    t_series_df_hor["power"],
                    label="Horizontal power",
                )
                # Plot slowness:
                ax[1].plot(
                    time_after_startZ,
                    t_series_df_Z["slowness"],
                    label="Vertical slowness",
                )
                ax[1].plot(
                    time_after_startH,
                    t_series_df_hor["slowness"],
                    label="Horizontal slowness",
                )
                # Plot back-azimuth:
                ax[2].plot(
                    time_after_startZ,
                    t_series_df_Z["back_azi"],
                    label="Vertical back-azimuth",
                )
                ax[2].plot(
                    time_after_startH,
                    t_series_df_hor["back_azi"],
                    label="Horizontal back-azimuth",
                )
                if len(events_df) > 0:
                    events_t1_after_start = (
                        np.array(events_df["t1"]) - t_series_df_Z["t"][0]
                    )
                    events_t2_after_start = (
                        np.array(events_df["t2"]) - t_series_df_Z["t"][0]
                    )
                    ax[0].scatter(
                        events_t1_after_start,
                        np.ones(len(events_df)) * np.max(t_series_df_Z["power"]),
                        c="r",
                        label="P phase picks",
                    )
                    ax[0].scatter(
                        events_t2_after_start,
                        np.ones(len(events_df)) * np.max(t_series_df_Z["power"]),
                        c="b",
                        label="S phase picks",
                    )
                else:
                    logger.info("No events to plot.")
                ax[0].legend()
                ax[2].set_xlabel("Time")
                ax[0].set_ylabel("Power (arb. units)")
                ax[1].set_ylabel("Slowness ($km$ $s^{-1}$)")
                ax[2].set_ylabel("Back-azimuth ($^o$)")
                fig.suptitle(
                    f"Beamforming time-series for {t_series_df_Z['t'][0]} to {t_series_df_Z['t'].values[-1]} UTC"
                )
                # plt.gca().yaxis.set_major_locator(MaxNLocator(5))
                for i in range(3):
                    ax[i].xaxis.set_major_locator(plt.MaxNLocator(3))
                figpath = Path(self.outdir, "plots", "detection_t_series")
                figpath.mkdir(parents=True, exist_ok=True)
                fig.savefig(f"{figpath}/Phase_association_{f_uid}.png", dpi=600)
                plt.show()

            # Append to datastore:
            events_df_all = pd.concat([events_df_all, events_df])
            events_df_all.reset_index(drop=True, inplace=True)
        return events_df_all

    def create_location_LUTs(
        self,
        oneD_vel_model_z_df,
        extent_x_m=4000,
        dxz=[100, 100],
        array_centre_xz=[0, 0],
    ):
        """Function to create lookup tables used for location. Lookup tables created are:
        - P travel-times
        - S travel-times.
        - P inclination angles.
        - S inclination angles.
        Note: Array centre is typically defined as the

        Parameters
        ----------
        oneD_vel_model_z_df : pandas DataFrame
            Pandas DataFrame containing 1D velocity model, with the following columns:
            depth | vp | vs
            All units are SI units (i.e. metres, metres/second). Depths are positive down.

        extent_x_m : float
            Extent of lookup table horizontal. Units are metres. Default is 4000 m.

        array_centre_xz : list
            Path to csv file containing station/receiver locations. Headers need
            to be of format: Latitude  Longitude  Elevation  Name. No need to change unless
            the user has a particularly good reason. Takes a list of length two, containing
            the array centre in x and z, in metres from the LUT grid origin.

        dxz : list
            List of two floats, defining the spatial spacing of the nodes in the LUTs in x
            and z. Units are metres. Default is [100, 100].

        Returns
        -------
        Returns a pickled LUT dict to <outdir>/LUT.
        """
        # Assign LUT creation parameters to class attributes:
        self.oneD_vel_model_z_df = oneD_vel_model_z_df
        self.extent_x_m = extent_x_m
        self.dxz = dxz
        self.array_centre_xz = array_centre_xz

        # Create 2D LUTs:
        # (for P and S travel-times, and inclination angle)
        # And get travel times:
        (
            trav_times_grid_P,
            trav_times_grid_S,
            theta_grid_P,
            theta_grid_S,
            vel_model_x_labels,
            vel_model_z_labels,
        ) = lookup_table_manager.create_2D_LUT(
            oneD_vel_model_z_df, array_centre_xz, extent_x_m=extent_x_m, dxz=dxz
        )
        # And save outputs:
        LUT_outdir = os.path.join(self.outdir, "LUT")
        os.makedirs(LUT_outdir, exist_ok=True)
        LUTs_dict = {}
        LUTs_dict["trav_times_grid_P"] = trav_times_grid_P
        LUTs_dict["trav_times_grid_S"] = trav_times_grid_S
        LUTs_dict["theta_grid_P"] = theta_grid_P
        LUTs_dict["theta_grid_S"] = theta_grid_S
        LUTs_dict["vel_model_x_labels"] = vel_model_x_labels
        LUTs_dict["vel_model_z_labels"] = vel_model_z_labels
        LUTs_dict["oneD_vel_model_z_df"] = self.oneD_vel_model_z_df
        LUTs_dict["extent_x_m"] = self.extent_x_m
        LUTs_dict["dxz"] = self.dxz
        LUTs_dict["array_centre_xz"] = self.array_centre_xz
        self.LUTs_fname = os.path.join(LUT_outdir, "LUTs.pkl")
        pickle.dump(LUTs_dict, open(self.LUTs_fname, "wb"))
        print("Saved LUT to:", self.LUTs_fname)
        self.LUTs_dict = LUTs_dict
        return LUTs_dict

    def load_location_LUTs(self, LUTs_fname=None):
        """Function to load lookup tables used for location.
        Parameters
        ----------
        LUTs_fname : str
            Path to LUT file to load. Optional. Default is to use the attribute
            <LUTs_fname>.
        """
        # Assign attribute, if not already specified correctly:
        if LUTs_fname:
            self.LUTs_fname = LUTs_fname
        # And load LUTs:
        LUTs_dict = pickle.load(open(self.LUTs_fname, "rb"))
        # And assign relevent attributes:
        self.oneD_vel_model_z_df = LUTs_dict["oneD_vel_model_z_df"]
        self.extent_x_m = LUTs_dict["extent_x_m"]
        self.dxz = LUTs_dict["dxz"]
        self.array_centre_xz = LUTs_dict["array_centre_xz"]
        self.LUTs_dict = LUTs_dict
        return LUTs_dict

    def locate_events(self, events_df, verbosity=0):
        """Function to locate events using LUT."""
        # Perform tests to check that various required attributes are specified:
        exit_bool = False
        if not self.LUTs_dict:
            print("Warning: LUTs_dict class attribute not specified. Exiting.")
            exit_bool = True
        if not self.array_latlon:
            print("Warning: array_latlon class attribute not specified. Exiting.")
            exit_bool = True
        if not self.receiver_vp:
            print("Warning: receiver_vp class attribute not specified. Exiting.")
            exit_bool = True
        if not self.receiver_vs:
            print("Warning: receiver_vs class attribute not specified. Exiting.")
            exit_bool = True

        # Locate events:
        if not exit_bool:
            events_df = location.locate_events_from_P_and_S_array_arrivals(
                events_df,
                self.LUTs_dict,
                self.array_latlon,
                self.receiver_vp,
                self.receiver_vs,
                verbosity=verbosity,
            )

        return events_df

    def save(self, out_fname=None):
        """Function to save class object to file.
        Parameters
        ----------
        out_fname : str
            Path to save class object to. Optional. If not specified, then save to
            <outdir>/detect_obj.pkl.
        """
        # Save class to file:
        if not out_fname:
            out_fname = os.path.join(self.outdir, "detect_obj.pkl")
        f = open(out_fname, "wb")
        pickle.dump(self.__dict__, f)
        f.close()
        print("Saved detection instance to:", out_fname)

    def load(self, preload_fname):
        """try load self.name.txt"""
        f = open(preload_fname, "rb")
        self.__dict__ = pickle.load(f)
        f.close()
        self.detector_type = self.__dict__.get("detector_type", "mad")
        self.selby_filter_half_width_hz = self.__dict__.get(
            "selby_filter_half_width_hz", 0.75
        )
        self.selby_filter_step_hz = self.__dict__.get("selby_filter_step_hz", 0.75)
        self.selby_noise_window_s = self.__dict__.get("selby_noise_window_s", 30.0)
        self.selby_f_threshold = self.__dict__.get("selby_f_threshold", 5.0)
        self.selby_f_prominence = self.__dict__.get("selby_f_prominence", None)
        self.selby_freqmin = self.__dict__.get("selby_freqmin", 0.5)
        self.selby_freqmax = self.__dict__.get("selby_freqmax", 6.5)
        print("Loaded detection instance from:", preload_fname)

    def get_composite_array_st_from_bazi_slowness(
        self,
        arrival_time,
        bazis_1_2,
        slows_1_2,
        t_before_s=10,
        t_after_s=10,
        st_out_fname="out.m",
        return_streams=False,
        method="linear",
        degree=1,
    ):
        """Function to find array stacked stream from back-azimuth and slowness. Returns average amplitude
        time-series seismogram of stacked array data, for all three componets.
        Parameters
        ----------
        arrival_time : obspy UTCDateTime object
            Arrival time at array. Will output window around this time, as defined by
            <t_before_s> and <t_after_s>.

        bazis_1_2 : list of 2 floats
            Back-azimuth of event phase arrivals in vertical and horizontal, in degrees from North.

        slows_1_2 : list of 2 floats
            Slowness of event phase arrivals in vertical and horizontal, in seconds/km.

        t_before_s : float
            Time, in seconds, before arrival time to include in window. Optional. Default
            is 10 s.

        t_after_s : float
            Time, in seconds, after arrival time to include in window. Optional. Default
            is 10 s.

        st_out_fname : str
            Filename of to save mseed data stream to. Optional. Default is out.m.

        return_streams : bool
            If True, returns st and composite_st. Optional. Default = False.
        """
        # Load in raw mseed data:
        st = self._load_data(
            arrival_time.year,
            arrival_time.month,
            arrival_time.day,
            hour=arrival_time.hour,
            norm=False,
        )
        # And trim data:
        st.trim(starttime=arrival_time - t_before_s, endtime=arrival_time + t_after_s)

        # Upsample data soas to provide best time shift later:
        st.interpolate(sampling_rate=10 * st[0].stats.sampling_rate)

        # Find and perform time shifts for all receivers:
        # Loop over stations:
        for i in range(len(st)):
            # Find time shift:
            # Get current station location (relative to array centre):
            curr_station = st[i].stats.station
            x_rec = self.stations_df.loc[self.stations_df["Name"] == curr_station][
                "x_array_coords_km"
            ].values[0]
            y_rec = self.stations_df.loc[self.stations_df["Name"] == curr_station][
                "y_array_coords_km"
            ].values[0]
            # Select either vertical or horizontal slowness and back-azimuth, depending on component:
            comp = st[i].stats.channel[-1]
            if comp == "Z":
                bazi = bazis_1_2[0]
                slow = slows_1_2[0]
            elif comp == "N" or comp == "E" or comp == "1" or comp == "2":
                bazi = bazis_1_2[1]
                slow = slows_1_2[1]
            # Calculate arrival time shift relative to centre of array:
            time_shift_curr_s = _calc_time_shift_from_array_cent(
                slow, bazi, x_rec, y_rec
            )

            # And perform time shift on data:
            n_samp_to_shift = round(time_shift_curr_s * st[i].stats.sampling_rate)
            st[i].data = np.roll(st[i].data, n_samp_to_shift)

        # And find stacked, mean and stdev of data:
        n_stat = len(st.select(channel="??Z"))
        # Get unique channels:
        chan_labels_tmp = []
        for tr in st:
            chan_labels_tmp.append(tr.stats.channel)
        # Create datastores to save to:
        max_st_len = 0
        for i in range(len(st)):
            if len(st[i].data) > max_st_len:
                max_st_len = len(st[i].data)
        Z_all = np.zeros((max_st_len, n_stat))
        N_all = np.zeros((max_st_len, n_stat))
        E_all = np.zeros((max_st_len, n_stat))

        # Loop over st, prepping data for stacking:
        for i in range(n_stat):
            # Z:
            if len(st.select(channel="??Z")[i].data) == max_st_len:
                Z_all[:, i] = st.select(channel="??Z")[i].data
            else:
                Z_all[: len(st.select(channel="??Z")[i].data), i] = st.select(
                    channel="??Z"
                )[i].data
            try:
                # N:
                if len(st.select(channel="??N")[i].data) == max_st_len:
                    N_all[:, i] = st.select(channel="??N")[i].data
                else:
                    N_all[: len(st.select(channel="??N")[i].data), i] = st.select(
                        channel="??N"
                    )[i].data
                # E:
                if len(st.select(channel="??E")[i].data) == max_st_len:
                    E_all[:, i] = st.select(channel="??E")[i].data
                else:
                    E_all[: len(st.select(channel="??E")[i].data), i] = st.select(
                        channel="??E"
                    )[i].data
            except IndexError:
                # And write if uses 1 and 2 labels rather than N and E:
                # N:
                if len(st.select(channel="??1")[i].data) == max_st_len:
                    N_all[:, i] = st.select(channel="??1")[i].data
                else:
                    N_all[: len(st.select(channel="??1")[i].data), i] = st.select(
                        channel="??1"
                    )[i].data
                # E:
                if len(st.select(channel="??2")[i].data) == max_st_len:
                    E_all[:, i] = st.select(channel="??2")[i].data
                else:
                    E_all[: len(st.select(channel="??2")[i].data), i] = st.select(
                        channel="??2"
                    )[i].data
            except:
                print("Date gap, continue... for now")
                continue
        # And create stacked data stream:
        if method == "linear":
            composite_st = _create_stacked_data_st(st, Z_all, N_all, E_all)
        elif method == "pws":
            composite_st = _create_phase_weighted_stack_st(
                st, Z_all, N_all, E_all, degree
            )
        elif method == "nth_root":
            pass
            # composite_st = _create_nth_root_stack_st(st, Z_all, N_all, E_all, degree)
        # And decimate data back down to original sampling rate:
        st.decimate(10, no_filter=True)
        composite_st.decimate(10, no_filter=True)

        # And save data:
        if st_out_fname:
            st.write(st_out_fname, format="MSEED")
            composite_st_out_fname = st_out_fname.split(".")[0] + "_composite.m"
            composite_st.write(composite_st_out_fname, format="MSEED")

        if return_streams:
            return st, composite_st
        else:
            del st, composite_st
            gc.collect()


def moving_window_mad(trace, window_len, mad_multiplier):
    """
    Function to calculate the median absolute deviation (MAD) using a
    moving window over a beam power time-series.

    For the start and end of the time-series, where a full window is not
    available, the window is truncated to fit within the time-series.

    Note that the MAD is scaled to be equivalent to the standard deviation
    for a Gaussian distribution, by multiplying by the constant 1.4826.
    See: https://en.wikipedia.org/wiki/Median_absolute_deviation

    Parameters
    ----------
    trace : np.ndarray
        1D numpy array containing beam power time-series.

    window_len : int
        Length of moving window, in number of samples.

    mad_multiplier : float
        Multiplier for MAD to set detection threshold. For example, a value of
        2 would set the threshold to be 2 times the MAD above the median. This is scaled
        by 1.4826 within the function to be equivalent to standard deviations for a
        Gaussian distribution.
    """
    # Create datastore:
    mad_thresholds = np.zeros(trace.shape)
    half_win = int(window_len / 2)
    for i in range(len(trace)):
        start = max(0, i - half_win)
        end = min(len(trace), i + half_win)
        window = trace[start:end]
        mad = np.median(np.abs(window - np.median(window)))
        # add one to mad_multiplier as the multiplier represents the
        # number of standard deviations above the median.
        mad_thresholds[i] = MAD_SCALE * (1 + mad_multiplier) * mad

    return mad_thresholds


# ------------- End: Define main functions --------------------
