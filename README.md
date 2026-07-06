# SeisSeeker

SeisSeeker is a Python package for array-based seismic detection and simple event location. It was originally developed for cryoseismology and icequake work, but the same workflow can be used for other small-aperture array datasets where you want:

- beamforming through time
- phase detection from array-derived traces
- simple P/S association
- approximate event locations from a 1D velocity model

The package currently exposes its workflow as Python classes and functions rather than a command-line interface, so the normal way to use it is from a script or notebook.

## What The Code Does

The main processing path lives in [`SeisSeeker/processing/detection.py`](SeisSeeker/processing/detection.py) and is built around the `setup_detection` class. A typical run is:

1. point SeisSeeker at a waveform archive and station list
2. run array processing to produce time-series CSV files
3. detect P and S arrivals from those time series
4. optionally build lookup tables and estimate event locations

There are currently two detector backends:

- `mad`: the original SeisSeeker workflow, using beam power time series plus MAD-based triggering
- `selby`: a newer Selby-inspired alternative backend that produces an F-like coherence score

The current `selby` backend is a practical approximation, not a full paper-faithful implementation of Selby's generalized `FMP` / `FMPAVE` detector.

## Installation

SeisSeeker does not currently declare its dependencies in `setup.py`, so the safest route is to create an environment and install the required packages explicitly.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas scipy matplotlib obspy numba scikit-fmm pytest
pip install -e .
```

### Dependency Notes

- `obspy` is required for waveform IO and time handling.
- `numba` is used for performance in the legacy beamforming path.
- `scikit-fmm` is required for lookup-table generation and, because of the current import structure, is effectively required for importing `SeisSeeker.processing`.
- `pytest` is only needed if you want to run the test suite.

## Dataset Requirements

### 1. Waveform Archive Layout

The current loader expects a day-based archive with files laid out as:

```text
<archive_root>/<YYYY>/<MM>/<DD>/<YYYYMMDDTHHMMSS>_<STATION>_<CHANNEL>.mseed
```

Example:

```text
/data/my_array/2026/07/06/20260706T130000_STA01_HHZ.mseed
```

Important:

- the file extension currently expected by the loader is `.mseed`
- the filename must contain timestamp, station code, and channel code exactly in that order
- channel patterns are matched from the values you pass in `channels_to_use`

The repository's bundled example archive under [`examples/rutford_icequake_example`](examples/rutford_icequake_example) uses `.msd` files, so if you want to rerun that example with the current code you will need to either:

- rename the example files to `.mseed`, or
- adapt `_load_data()` in [`SeisSeeker/processing/detection.py`](SeisSeeker/processing/detection.py)

### 2. Station File

The station CSV must contain these columns:

```text
Latitude,Longitude,Elevation,Name
```

Example:

```csv
Latitude,Longitude,Elevation,Name
-78.1456985294,-83.9369028595,321.67,A000
-78.1457087418,-83.9360617647,321.67,AS11
```

See [`examples/rutford_icequake_example/inputs/AS_stations.txt`](examples/rutford_icequake_example/inputs/AS_stations.txt) for a real example.

### 3. Channel Conventions

`channels_to_use` must currently be one of:

- `["??Z"]` for vertical-only processing
- `["??Z", "??N", "??E"]` for Z/N/E component processing
- `["??Z", "??1", "??2"]` for Z/1/2 component processing

If you want event detection and phase association, use three components.

## Quick Start: Detection On A New Dataset

This is the smallest practical end-to-end example for detection.

```python
import obspy
from SeisSeeker.processing.detection import setup_detection

det = setup_detection(
    archivedir="/path/to/archive",
    outdir="/path/to/output",
    stations_fname="/path/to/stations.csv",
    starttime=obspy.UTCDateTime("2026-07-06T00:00:00"),
    endtime=obspy.UTCDateTime("2026-07-06T06:00:00"),
    channels_to_use=["??Z", "??N", "??E"],
)

# Beamforming / array-processing settings
det.freqmin = 1.0
det.freqmax = 15.0
det.num_freqs = 80
det.min_sl = 0.0
det.max_sl = 1.0
det.n_sl = 51
det.min_baz = 0.0
det.max_baz = 360.0
det.n_baz = 181
det.win_len_s = 0.2
det.win_step_inc_s = 0.1
det.remove_autocorr = True

# Original MAD detector settings
det.detector_type = "mad"
det.mad_window_length_s = 1800
det.mad_multiplier = 8
det.min_event_sep_s = 1.0
det.max_phase_sep_s = 2.5
det.bazi_tol = 20.0

det.run_array_proc()
events_df = det.detect_events()
det.save()

print(events_df.head())
```

### Outputs Produced

`run_array_proc()` writes per-component CSV time series into `outdir`, for example:

- `detection_t_series_20260706_1300_chZ.csv`
- `detection_t_series_20260706_1300_chN.csv`
- `detection_t_series_20260706_1300_chE.csv`

Each file contains:

- `t`
- `power`
- `score`
- `slowness`
- `back_azi`
- `statistic`

`detect_events()` returns a `pandas.DataFrame` of phase-associated events in memory. You can save it yourself with `events_df.to_csv(...)` if needed.

## Using The Selby-Inspired Backend

If you want the newer alternative detector backend, switch `detector_type` to `selby` and set the Selby-specific controls:

```python
det.detector_type = "selby"
det.selby_freqmin = 0.5
det.selby_freqmax = 6.5
det.selby_filter_step_hz = 0.75
det.selby_filter_half_width_hz = 0.75
det.selby_noise_window_s = 30.0
det.selby_f_threshold = 5.0
det.selby_f_prominence = 5.0
```

In this mode:

- output files get a `_selby` suffix
- the detector triggers on the `score` column rather than MAD on beam power
- uncertainty estimation through the old beam-power workflow is not implemented

## Event Location Workflow

After detection, SeisSeeker can estimate simple locations using:

- the P-S lag
- apparent slowness-derived incidence angle
- a 2D lookup table built from a 1D velocity model

Example:

```python
import pandas as pd

vel_model = pd.read_csv("/path/to/1D_velocity_model.csv")

det.create_location_LUTs(
    oneD_vel_model_z_df=vel_model,
    extent_x_m=4000,
    dxz=[100, 100],
    array_centre_xz=[0, 0],
)

det.array_latlon = (-78.1457, -83.9369)
det.receiver_vp = 3.8
det.receiver_vs = 1.95

located_df = det.locate_events(events_df)
print(located_df.head())
```

### Location Input Units

This part is easy to get wrong:

- the velocity-model table used by `create_location_LUTs()` is expected in metres and metres/second
- the `receiver_vp` and `receiver_vs` attributes used by `locate_events()` should be in km/s

That mixed-unit behavior reflects the current implementation, so it is worth checking carefully when preparing a new dataset.

## Minimal Checklist For A New Dataset

Before running SeisSeeker on a fresh archive, check:

1. waveform files are in the expected directory tree
2. waveform filenames end in `.mseed`
3. station names in filenames match the `Name` column in the station CSV
4. channels on disk match the `channels_to_use` patterns
5. start/end times fall inside the archive coverage
6. the chosen frequency band makes sense for your target signals
7. if locating events, your 1D velocity model has `depth`, `vp`, and `vs` columns

## Example Data In This Repository

The repository contains a worked example area under [`examples/rutford_icequake_example`](examples/rutford_icequake_example), including:

- station files
- a 1D velocity model
- an example waveform archive
- previously generated output CSV files

The notebook in that directory is useful for orientation, but the example archive naming does not exactly match the current `.mseed` loader expectation.

## Running Tests

The repository now includes a small test suite for the Selby-inspired backend:

```bash
pytest -q tests/test_selby.py
```

## Current Limitations

- no CLI wrapper yet; usage is through Python
- dependencies are not yet declared in `setup.py`
- the top-level import path is stricter than it needs to be because `processing` imports the LUT code eagerly
- the Selby backend is an approximation, not a full implementation of Selby 2008/2011/2013
- the example archive extension and the live loader expectation currently differ

## Citation / Background

Background documentation for the original array-processing approach is referenced in the project history:

Thomas S. Hudson, Alex M. Brisbourne, Sofia-Katerina Kufner, J-Michael Kendall, and Andy M. Smith, "Array processing in cryoseismology".

