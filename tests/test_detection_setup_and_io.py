import datetime

import numpy as np


def test_setup_detection_initialises_station_geometry(make_detector):
    detection, detector = make_detector()

    assert isinstance(detector, detection.setup_detection)
    assert detector.detector_type == "mad"
    assert detector.channels_to_use == ["??Z", "??N", "??E"]
    assert "x_array_coords_km" in detector.stations_df.columns
    assert "theta_array_coords_deg" in detector.stations_df.columns
    np.testing.assert_allclose(detector.stations_df["x_array_coords_km"].mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(detector.stations_df["y_array_coords_km"].mean(), 0.0, atol=1e-12)


def test_detector_suffix_outfile_name_and_prefilter_band(make_detector):
    _, detector = make_detector()
    date = datetime.date(2020, 1, 1)

    assert detector._detector_suffix() == ""
    assert detector._build_outfile_name(date, 3, "??Z") == "detection_t_series_20200101_0300_chZ.csv"
    assert detector._get_active_prefilter_band() == (None, None)

    detector.detector_type = "selby"
    detector.selby_freqmin = 0.5
    detector.selby_freqmax = 6.5
    assert detector._detector_suffix() == "_selby"
    assert (
        detector._build_outfile_name(date, 4, "??N")
        == "detection_t_series_20200101_0400_chN_selby.csv"
    )
    assert detector._get_active_prefilter_band() == (0.5, 6.5)


def test_load_data_reads_filters_and_trims_archive(
    make_detector, make_trace, write_archive, monkeypatch
):
    sampling_rate = 20.0
    t = np.arange(int(10 * sampling_rate)) / sampling_rate
    waveform = np.sin(2.0 * np.pi * 1.5 * t) + 0.2 * np.sin(2.0 * np.pi * 6.0 * t)
    traces = [
        make_trace("STA01", "HHZ", waveform, sampling_rate=sampling_rate),
        make_trace("STA02", "HHZ", waveform * 0.5, sampling_rate=sampling_rate),
        make_trace("STA03", "HHZ", waveform * 0.25, sampling_rate=sampling_rate),
    ]
    archive_root = write_archive(traces)
    detection, detector = make_detector(
        archivedir=archive_root,
        starttime="2020-01-01T00:00:01",
        endtime="2020-01-01T00:00:05",
        channels_to_use=["??Z"],
    )
    detector.freqmin = 1.0
    detector.freqmax = 3.0

    traces_by_station = {
        trace.stats.station: trace.copy()
        for trace in traces
    }

    def fake_read(pattern):
        station = pattern.split("_")[-2]
        return detection.obspy.Stream([traces_by_station[station].copy()])

    monkeypatch.setattr(detection.obspy, "read", fake_read)

    stream = detector._load_data(2020, 1, 1, hour=0)

    assert len(stream) == 3
    assert all(trace.stats.starttime >= detector.starttime for trace in stream)
    assert all(trace.stats.endtime <= detector.endtime for trace in stream)
    assert all(trace.stats.channel == "HHZ" for trace in stream)
    assert np.max(np.abs(stream[0].data)) > 0.0


def test_save_and_load_round_trip_detector_settings(make_detector, tmp_path):
    _, detector = make_detector()
    detector.detector_type = "selby"
    detector.selby_filter_half_width_hz = 0.5
    detector.selby_filter_step_hz = 0.25
    detector.selby_noise_window_s = 12.0
    detector.selby_f_threshold = 4.0
    detector.selby_f_prominence = 3.0
    detector.selby_freqmin = 0.8
    detector.selby_freqmax = 7.2
    detector.mad_multiplier = 11

    outpath = tmp_path / "detector.pkl"
    detector.save(str(outpath))

    _, restored = make_detector(outdir=tmp_path / "restored_outputs")
    restored.load(str(outpath))

    assert restored.detector_type == "selby"
    assert restored.selby_filter_half_width_hz == 0.5
    assert restored.selby_filter_step_hz == 0.25
    assert restored.selby_noise_window_s == 12.0
    assert restored.selby_f_threshold == 4.0
    assert restored.selby_f_prominence == 3.0
    assert restored.selby_freqmin == 0.8
    assert restored.selby_freqmax == 7.2
    assert restored.mad_multiplier == 11
