import importlib
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import obspy
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
PROCESSING_DIR = ROOT / "SeisSeeker" / "processing"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def configure_test_environment(monkeypatch, tmp_path):
    """Provide stable import and cache paths for detector tests."""
    mpl_dir = tmp_path / "mplconfig"
    mpl_dir.mkdir()
    monkeypatch.setenv("MPLCONFIGDIR", str(mpl_dir))
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_dir))
    monkeypatch.syspath_prepend(str(ROOT))

    fake_skfmm = types.ModuleType("skfmm")

    def _unused_travel_time(*args, **kwargs):  # pragma: no cover - guard rail only
        raise NotImplementedError("skfmm travel_time is outside detector-test scope.")

    fake_skfmm.travel_time = _unused_travel_time
    monkeypatch.setitem(sys.modules, "skfmm", fake_skfmm)

    fake_numba = types.ModuleType("numba")

    def _jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    @contextmanager
    def _objmode(**kwargs):
        yield

    def _prange(*args):
        return range(*args)

    def _set_num_threads(*args, **kwargs):
        return None

    fake_numba.jit = _jit
    fake_numba.objmode = _objmode
    fake_numba.prange = _prange
    fake_numba.set_num_threads = _set_num_threads
    monkeypatch.setitem(sys.modules, "numba", fake_numba)


@pytest.fixture
def load_processing_module():
    """Load a processing module directly from its source file."""

    def _load(module_name):
        return importlib.import_module(f"SeisSeeker.processing.{module_name}")

    return _load


@pytest.fixture
def station_records():
    return [
        {
            "Latitude": 64.0000,
            "Longitude": -21.0000,
            "Elevation": 100.0,
            "Name": "STA01",
        },
        {
            "Latitude": 64.0000,
            "Longitude": -20.9990,
            "Elevation": 120.0,
            "Name": "STA02",
        },
        {
            "Latitude": 64.0010,
            "Longitude": -21.0000,
            "Elevation": 90.0,
            "Name": "STA03",
        },
    ]


@pytest.fixture
def write_station_csv(tmp_path):
    def _write(records):
        path = tmp_path / "stations.csv"
        pd.DataFrame(records).to_csv(path, index=False)
        return path

    return _write


@pytest.fixture
def station_csv(write_station_csv, station_records):
    return write_station_csv(station_records)


@pytest.fixture
def make_trace():
    def _make(
        station,
        channel,
        data,
        starttime="2020-01-01T00:00:00",
        sampling_rate=20.0,
    ):
        trace = obspy.Trace(np.asarray(data, dtype=np.float32).copy())
        trace.stats.station = station
        trace.stats.channel = channel
        trace.stats.starttime = obspy.UTCDateTime(starttime)
        trace.stats.sampling_rate = sampling_rate
        return trace

    return _make


@pytest.fixture
def make_stream():
    def _make(traces):
        return obspy.Stream(traces=[trace.copy() for trace in traces])

    return _make


@pytest.fixture
def write_archive(tmp_path):
    def _write(traces):
        archive_root = tmp_path / "archive"
        for trace in traces:
            starttime = trace.stats.starttime
            day_dir = (
                archive_root
                / f"{starttime.year:04d}"
                / f"{starttime.month:02d}"
                / f"{starttime.day:02d}"
            )
            day_dir.mkdir(parents=True, exist_ok=True)
            timestamp = (
                f"{starttime.year:04d}{starttime.month:02d}{starttime.day:02d}"
                f"T{starttime.hour:02d}{starttime.minute:02d}{starttime.second:02d}"
            )
            outpath = day_dir / f"{timestamp}_{trace.stats.station}_{trace.stats.channel}.mseed"
            obspy.Stream([trace.copy()]).write(str(outpath), format="MSEED")
        return archive_root

    return _write


@pytest.fixture
def make_score_cube():
    def _make(n_windows=3, n_sl=4, n_baz=5, fill_value=0.0, peaks=None):
        cube = np.full((n_windows, n_sl, n_baz), fill_value, dtype=float)
        for win_idx, sl_idx, baz_idx, value in peaks or []:
            cube[win_idx, sl_idx, baz_idx] = value
        return cube

    return _make


@pytest.fixture
def make_detector(load_processing_module, station_csv, tmp_path):
    def _make(
        archivedir=None,
        outdir=None,
        starttime="2020-01-01T00:00:00",
        endtime="2020-01-01T00:02:00",
        channels_to_use=None,
    ):
        detection = load_processing_module("detection")
        archive_root = Path(archivedir or (tmp_path / "archive"))
        archive_root.mkdir(parents=True, exist_ok=True)
        outdir_path = Path(outdir or (tmp_path / "outputs"))
        outdir_path.mkdir(parents=True, exist_ok=True)
        detector = detection.setup_detection(
            archivedir=str(archive_root),
            outdir=str(outdir_path),
            stations_fname=str(station_csv),
            starttime=obspy.UTCDateTime(starttime),
            endtime=obspy.UTCDateTime(endtime),
            channels_to_use=channels_to_use or ["??Z", "??N", "??E"],
        )
        return detection, detector

    return _make
