from pathlib import Path

import numpy as np
import obspy
import pandas as pd
import pytest


def _write_detection_csv(outdir, uid, channel_token, rows, suffix=""):
    path = Path(outdir) / f"detection_t_series_{uid}_ch{channel_token}{suffix}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _make_rows(times, power, score, slowness, back_azi, statistic):
    return {
        "t": [str(time) for time in times],
        "power": power,
        "score": score,
        "slowness": slowness,
        "back_azi": back_azi,
        "statistic": [statistic] * len(power),
    }


def test_run_array_proc_writes_expected_csvs_for_mad(
    make_detector, make_trace, make_stream, make_score_cube, monkeypatch
):
    _, detector = make_detector(
        starttime="2020-01-01T00:00:00",
        endtime="2020-01-01T00:00:20",
    )
    score_cube = make_score_cube(
        n_windows=2,
        n_sl=3,
        n_baz=4,
        peaks=[(0, 1, 2, 4.0), (1, 2, 1, 5.0)],
        fill_value=0.5,
    )
    stream = make_stream(
        [make_trace("STA01", "HHZ", np.arange(400), sampling_rate=20.0)]
    )

    monkeypatch.setattr(detector, "_load_data", lambda **kwargs: stream.copy())
    monkeypatch.setattr(detector, "_beamforming", lambda st_trimmed: score_cube.copy())

    detector.run_array_proc()

    outfiles = sorted(Path(detector.outdir).glob("detection_t_series_*_ch*.csv"))
    assert [path.name for path in outfiles] == [
        "detection_t_series_20200101_0000_chE.csv",
        "detection_t_series_20200101_0000_chN.csv",
        "detection_t_series_20200101_0000_chZ.csv",
    ]
    store_df = pd.read_csv(outfiles[-1])
    assert list(store_df.columns) == [
        "t",
        "power",
        "score",
        "slowness",
        "back_azi",
        "statistic",
    ]
    np.testing.assert_allclose(store_df["power"], [4.0, 5.0])
    np.testing.assert_allclose(store_df["score"], [4.0, 5.0])
    assert set(store_df["statistic"]) == {"beam_power"}
    assert len(detector.out_fnames_array_proc) == 3


def test_run_array_proc_writes_selby_suffix(make_detector, make_trace, make_stream, make_score_cube, monkeypatch):
    _, detector = make_detector(
        starttime="2020-01-01T00:00:00",
        endtime="2020-01-01T00:00:20",
        channels_to_use=["??Z"],
    )
    detector.detector_type = "selby"
    score_cube = make_score_cube(
        n_windows=2,
        n_sl=2,
        n_baz=3,
        peaks=[(0, 0, 1, 6.0), (1, 1, 2, 8.0)],
        fill_value=0.25,
    )
    stream = make_stream(
        [make_trace("STA01", "HHZ", np.arange(400), sampling_rate=20.0)]
    )

    monkeypatch.setattr(detector, "_load_data", lambda **kwargs: stream.copy())
    monkeypatch.setattr(
        detector, "_selby_beamforming", lambda st_trimmed: score_cube.copy()
    )

    detector.run_array_proc()

    outfile = Path(detector.outdir) / "detection_t_series_20200101_0000_chZ_selby.csv"
    assert outfile.is_file()
    store_df = pd.read_csv(outfile)
    np.testing.assert_allclose(store_df["score"], [6.0, 8.0])
    assert set(store_df["statistic"]) == {"selby_f"}


def test_detect_events_mad_uses_beam_power_thresholds(make_detector, monkeypatch):
    detection, detector = make_detector()
    start = obspy.UTCDateTime("2020-01-01T00:00:00")
    uid = "20200101_0000"
    rows_z = _make_rows(
        [start + i for i in range(5)],
        power=[0.2, 5.0, 0.2, 0.2, 0.2],
        score=[0.2, 0.2, 0.2, 0.2, 0.2],
        slowness=[0.1, 0.2, 0.2, 0.2, 0.2],
        back_azi=[20.0, 30.0, 30.0, 30.0, 30.0],
        statistic="beam_power",
    )
    rows_n = _make_rows(
        [start + i for i in range(5)],
        power=[0.2, 0.2, 4.0, 0.2, 0.2],
        score=[0.1] * 5,
        slowness=[0.2] * 5,
        back_azi=[31.0] * 5,
        statistic="beam_power",
    )
    rows_e = _make_rows(
        [start + i for i in range(5)],
        power=[0.2, 0.2, 3.0, 0.2, 0.2],
        score=[0.1] * 5,
        slowness=[0.2] * 5,
        back_azi=[29.0] * 5,
        statistic="beam_power",
    )

    z_path = _write_detection_csv(detector.outdir, uid, "Z", rows_z)
    _write_detection_csv(detector.outdir, uid, "N", rows_n)
    _write_detection_csv(detector.outdir, uid, "E", rows_e)
    detector.out_fnames_array_proc = [str(z_path)]
    detector.min_event_sep_s = 1.0
    detector.max_phase_sep_s = 2.0
    detector.bazi_tol = 5.0
    monkeypatch.setattr(
        detection, "moving_window_mad", lambda trace, window_len, mad_multiplier: np.ones_like(trace)
    )

    events_df = detector.detect_events()

    assert len(events_df) == 1
    assert events_df.iloc[0]["pow1"] == 5.0
    assert events_df.iloc[0]["pow2"] > 4.9


def test_detect_events_selby_uses_score_column(make_detector):
    _, detector = make_detector()
    start = obspy.UTCDateTime("2020-01-01T00:00:00")
    uid = "20200101_0000"
    detector.detector_type = "selby"
    detector.selby_f_threshold = 3.0
    detector.selby_f_prominence = 3.0

    rows_z = _make_rows(
        [start + i for i in range(5)],
        power=[0.1] * 5,
        score=[0.1, 5.0, 0.2, 0.2, 0.2],
        slowness=[0.1, 0.2, 0.2, 0.2, 0.2],
        back_azi=[40.0] * 5,
        statistic="selby_f",
    )
    rows_n = _make_rows(
        [start + i for i in range(5)],
        power=[0.1] * 5,
        score=[0.1, 0.2, 4.5, 0.2, 0.2],
        slowness=[0.2] * 5,
        back_azi=[42.0] * 5,
        statistic="selby_f",
    )
    rows_e = _make_rows(
        [start + i for i in range(5)],
        power=[0.1] * 5,
        score=[0.1, 0.2, 3.5, 0.2, 0.2],
        slowness=[0.2] * 5,
        back_azi=[38.0] * 5,
        statistic="selby_f",
    )

    z_path = _write_detection_csv(detector.outdir, uid, "Z", rows_z, suffix="_selby")
    _write_detection_csv(detector.outdir, uid, "N", rows_n, suffix="_selby")
    _write_detection_csv(detector.outdir, uid, "E", rows_e, suffix="_selby")
    detector.out_fnames_array_proc = [str(z_path)]

    events_df = detector.detect_events()

    assert len(events_df) == 1
    assert str(events_df.iloc[0]["t1"]) == str(start + 1)
    assert str(events_df.iloc[0]["t2"]) == str(start + 2)


def test_detect_events_falls_back_to_ch1_ch2_and_trims_shortest_input(make_detector, monkeypatch):
    detection, detector = make_detector()
    start = obspy.UTCDateTime("2020-01-01T00:00:00")
    uid = "20200101_0000"
    rows_z = _make_rows(
        [start + i for i in range(6)],
        power=[0.2, 5.0, 0.2, 0.2, 0.2, 0.2],
        score=[0.2] * 6,
        slowness=[0.2] * 6,
        back_azi=[50.0] * 6,
        statistic="beam_power",
    )
    rows_1 = _make_rows(
        [start + i for i in range(5)],
        power=[0.2, 0.2, 4.0, 0.2, 0.2],
        score=[0.1] * 5,
        slowness=[0.2] * 5,
        back_azi=[52.0] * 5,
        statistic="beam_power",
    )
    rows_2 = _make_rows(
        [start + i for i in range(4)],
        power=[0.2, 0.2, 3.0, 0.2],
        score=[0.1] * 4,
        slowness=[0.2] * 4,
        back_azi=[48.0] * 4,
        statistic="beam_power",
    )

    z_path = _write_detection_csv(detector.outdir, uid, "Z", rows_z)
    _write_detection_csv(detector.outdir, uid, "1", rows_1)
    _write_detection_csv(detector.outdir, uid, "2", rows_2)
    detector.out_fnames_array_proc = [str(z_path)]
    monkeypatch.setattr(
        detection, "moving_window_mad", lambda trace, window_len, mad_multiplier: np.ones_like(trace)
    )
    monkeypatch.setattr(detection.logger, "warning", lambda *args, **kwargs: None)

    events_df = detector.detect_events()

    assert len(events_df) == 1
    assert str(events_df.iloc[0]["t2"]) == str(start + 2)


def test_detect_events_selby_uncertainty_request_raises(make_detector):
    _, detector = make_detector()
    start = obspy.UTCDateTime("2020-01-01T00:00:00")
    uid = "20200101_0000"
    detector.detector_type = "selby"
    detector.selby_f_threshold = 3.0
    detector.calc_uncertainties = True

    rows_z = _make_rows(
        [start + i for i in range(5)],
        power=[0.1] * 5,
        score=[0.1, 5.0, 0.2, 0.2, 0.2],
        slowness=[0.2] * 5,
        back_azi=[60.0] * 5,
        statistic="selby_f",
    )
    rows_n = _make_rows(
        [start + i for i in range(5)],
        power=[0.1] * 5,
        score=[0.1, 0.2, 4.5, 0.2, 0.2],
        slowness=[0.2] * 5,
        back_azi=[61.0] * 5,
        statistic="selby_f",
    )
    rows_e = _make_rows(
        [start + i for i in range(5)],
        power=[0.1] * 5,
        score=[0.1, 0.2, 3.5, 0.2, 0.2],
        slowness=[0.2] * 5,
        back_azi=[59.0] * 5,
        statistic="selby_f",
    )

    z_path = _write_detection_csv(detector.outdir, uid, "Z", rows_z, suffix="_selby")
    _write_detection_csv(detector.outdir, uid, "N", rows_n, suffix="_selby")
    _write_detection_csv(detector.outdir, uid, "E", rows_e, suffix="_selby")
    detector.out_fnames_array_proc = [str(z_path)]

    with pytest.raises(NotImplementedError):
        detector.detect_events()


def test_detect_events_returns_empty_dataframe_when_no_events(make_detector, monkeypatch):
    detection, detector = make_detector()
    start = obspy.UTCDateTime("2020-01-01T00:00:00")
    uid = "20200101_0000"
    rows = _make_rows(
        [start + i for i in range(5)],
        power=[0.2] * 5,
        score=[0.2] * 5,
        slowness=[0.2] * 5,
        back_azi=[10.0] * 5,
        statistic="beam_power",
    )

    z_path = _write_detection_csv(detector.outdir, uid, "Z", rows)
    _write_detection_csv(detector.outdir, uid, "N", rows)
    _write_detection_csv(detector.outdir, uid, "E", rows)
    detector.out_fnames_array_proc = [str(z_path)]
    monkeypatch.setattr(
        detection,
        "moving_window_mad",
        lambda trace, window_len, mad_multiplier: np.ones_like(trace) * 10.0,
    )

    events_df = detector.detect_events()

    assert events_df.empty
