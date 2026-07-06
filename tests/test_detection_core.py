from types import SimpleNamespace

import numpy as np
import obspy
import pandas as pd


def test_flatten_list_and_xy_to_rtheta(load_processing_module):
    detection = load_processing_module("detection")

    assert detection.flatten_list([[1, 2], [3], [], [4, 5]]) == [1, 2, 3, 4, 5]

    radii, thetas = detection.xy_to_rtheta(np.array([1.0, 0.0]), np.array([0.0, -1.0]))
    np.testing.assert_allclose(radii, [1.0, 1.0])
    np.testing.assert_allclose(thetas, [90.0, 180.0])

    radius, theta = detection.xy_to_rtheta(-1.0, 0.0)
    assert radius == 1.0
    assert theta == 270.0


def test_ensure_score_columns_backfills_but_does_not_overwrite(load_processing_module):
    detection = load_processing_module("detection")

    t_series_df = pd.DataFrame({"power": [2.0, 3.0]})
    updated = detection._ensure_score_columns(t_series_df.copy())
    np.testing.assert_allclose(updated["score"], [2.0, 3.0])
    assert list(updated["statistic"]) == ["beam_power", "beam_power"]

    prefilled = pd.DataFrame(
        {"power": [1.0], "score": [9.0], "statistic": ["selby_f"]}
    )
    preserved = detection._ensure_score_columns(prefilled.copy())
    assert preserved.loc[0, "score"] == 9.0
    assert preserved.loc[0, "statistic"] == "selby_f"


def test_calc_time_shift_find_max_power_and_moving_window_mad(load_processing_module):
    detection = load_processing_module("detection")

    time_shift = detection._calc_time_shift_from_array_cent(
        slow=0.25, bazi=90.0, x_rec=2.0, y_rec=1.0
    )
    assert np.isclose(time_shift, 0.5)

    events = [
        SimpleNamespace(pow1=1.0, pow2=2.0),
        SimpleNamespace(pow1=3.0, pow2=4.0),
        SimpleNamespace(pow1=2.0, pow2=2.0),
    ]
    assert detection._find_max_power_event(events) is events[1]

    trace = np.array([1.0, 1.0, 2.0, 10.0, 2.0])
    thresholds = detection.moving_window_mad(trace, window_len=4, mad_multiplier=1.0)
    np.testing.assert_allclose(
        thresholds,
        np.array([0.0, 0.0, 1.4826, 1.4826, 0.0]),
        rtol=1e-6,
    )


def test_phase_associator_core_worker_respects_time_and_bazi(load_processing_module):
    detection = load_processing_module("detection")

    matches = detection._phase_associator_core_worker(
        peaks_Z=np.array([0, 1]),
        peaks_hor=np.array([0, 1, 2]),
        bazis_Z=np.array([10.0, 80.0]),
        bazis_hor=np.array([14.0, 120.0, 85.0]),
        bazi_tol=8.0,
        t_Z_secs_after_start=np.array([0.0, 2.0]),
        t_hor_secs_after_start=np.array([0.5, 2.1, 2.7]),
        max_phase_sep_s=1.0,
        min_phase_sep_s=0.2,
    )

    assert matches == [[0, 0], [1, 2]]


def test_phase_associator_filters_to_max_power_event(load_processing_module):
    detection = load_processing_module("detection")
    start = obspy.UTCDateTime("2020-01-01T00:00:00")

    t_series_df_Z = pd.DataFrame(
        {
            "t": [start + i for i in range(4)],
            "power": [0.5, 2.0, 6.0, 0.2],
            "slowness": [0.1, 0.2, 0.3, 0.4],
            "back_azi": [20.0, 20.0, 22.0, 22.0],
        }
    )
    t_series_df_hor = pd.DataFrame(
        {
            "t": [start, start + 1.4, start + 2.4, start + 3.4],
            "power": [0.5, 1.5, 7.0, 0.2],
            "slowness": [0.1, 0.2, 0.3, 0.4],
            "back_azi": [21.0, 21.0, 23.0, 23.0],
        }
    )

    events_df = detection._phase_associator(
        t_series_df_Z=t_series_df_Z,
        t_series_df_hor=t_series_df_hor,
        peaks_Z=np.array([1, 2]),
        peaks_hor=np.array([1, 2]),
        bazi_tol=5.0,
        filt_phase_assoc_by_max_power=True,
        max_phase_sep_s=1.0,
        min_phase_sep_s=0.2,
        min_event_sep_s=2.0,
    )

    assert len(events_df) == 1
    assert events_df.iloc[0]["pow1"] == 6.0
    assert events_df.iloc[0]["pow2"] == 7.0
    assert events_df.iloc[0]["t1"] == start + 2
    assert events_df.iloc[0]["t2"] == start + 2.4


def test_stack_results_and_find_time_series(make_detector):
    detection, detector = make_detector()

    pfreq_all = np.zeros((1, 2, 1, 1), dtype=np.complex128)
    pfreq_all[0, 0, 0, 0] = 3.0 + 0j
    pfreq_all[0, 1, 0, 0] = 1.0 + 0j
    summed = detector._stack_results(pfreq_all)
    np.testing.assert_allclose(summed[0, 0, 0], 4.0)

    detector.norm_pre_stacking = True
    normed = detector._stack_results(pfreq_all)
    np.testing.assert_allclose(normed[0, 0, 0], 1.0)

    detector.min_sl = 0.0
    detector.max_sl = 0.3
    detector.min_baz = 0.0
    detector.max_baz = 360.0
    detector.win_step_inc_s = 0.5
    score_cube = np.zeros((2, 3, 4))
    score_cube[0, 1, 2] = 5.0
    score_cube[1, 2, 1] = 7.0

    t_series, powers, slownesses, back_azis = detector._find_time_series(score_cube)
    np.testing.assert_allclose(t_series, [0.25, 0.75])
    np.testing.assert_allclose(powers, [5.0, 7.0])
    np.testing.assert_allclose(slownesses, [0.15, 0.3])
    np.testing.assert_allclose(back_azis, [180.0, 90.0])


def test_convert_st_to_station_matrix_fills_missing_station(make_detector, make_trace, make_stream):
    _, detector = make_detector()
    detector.channel_curr = "??Z"

    stream = make_stream(
        [
            make_trace("STA01", "HHZ", [1, 2, 3, 4]),
            make_trace("STA03", "HHZ", [5, 6]),
        ]
    )
    matrix = detector._convert_st_to_station_matrix(stream)

    assert matrix.shape == (3, 4)
    np.testing.assert_allclose(matrix[0], [1, 2, 3, 4])
    np.testing.assert_allclose(matrix[1], [0, 0, 0, 0])
    np.testing.assert_allclose(matrix[2], [5, 6, 0, 0])


def test_convert_st_to_np_data_zero_pads_short_windows(make_detector, make_trace, make_stream):
    _, detector = make_detector()
    detector.channel_curr = "??Z"
    detector.win_pad_s = 0.0
    detector.win_step_inc_s = 0.5
    detector.win_len_s = 1.0

    stream = make_stream(
        [
            make_trace("STA01", "HHZ", [1, 2, 3, 4, 5, 6], sampling_rate=4.0),
            make_trace("STA02", "HHZ", [10, 11, 12, 13, 14, 15], sampling_rate=4.0),
        ]
    )
    data = detector._convert_st_to_np_data(stream)

    assert data.shape == (2, 3, 4)
    np.testing.assert_allclose(data[0, 0], [1, 2, 3, 4])
    np.testing.assert_allclose(data[0, 1], [10, 11, 12, 13])
    np.testing.assert_allclose(data[:, 2], 0.0)
    np.testing.assert_allclose(data[1], 0.0)


def test_fast_freq_domain_array_proc_returns_complex_cube(load_processing_module):
    detection = load_processing_module("detection")
    data = np.array([[[0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0]] * 2])

    cube = detection._fast_freq_domain_array_proc(
        data=data,
        min_sl=0.0,
        max_sl=0.2,
        n_sl=3,
        min_baz=0.0,
        max_baz=180.0,
        n_baz=4,
        fs=8.0,
        target_freqs=np.array([1.0, 2.0]),
        xx=np.array([0.0, 0.5]),
        yy=np.array([0.0, 0.0]),
        n_stations=2,
        n_t_samp=8,
        remove_autocorr=False,
    )

    assert cube.shape == (1, 2, 3, 4)
    assert np.iscomplexobj(cube)
    assert np.all(np.isfinite(cube.real))
