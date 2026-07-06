import numpy as np


def _shift_signal(signal, dt_s, fs):
    time = np.arange(len(signal)) / fs
    shifted_time = time + dt_s
    return np.interp(shifted_time, time, signal, left=0.0, right=0.0)


def _make_burst_signal(fs, duration_s, centre_freq_hz):
    time = np.arange(int(duration_s * fs)) / fs
    signal = np.zeros_like(time)
    mask = (time >= 2.0) & (time <= 4.0)
    taper = np.hanning(mask.sum())
    signal[mask] = taper * np.sin(2.0 * np.pi * centre_freq_hz * time[mask])
    return signal


def test_build_selby_filter_centres_creates_overlapping_band_centres(load_processing_module):
    selby = load_processing_module("selby")
    centres = selby.build_selby_filter_centres(0.5, 6.5, 0.75, 0.75)
    assert np.all(centres >= 1.25)
    assert np.all(centres <= 5.75)
    assert len(centres) > 1


def test_rolling_noise_floor_uses_trailing_medians(load_processing_module):
    selby = load_processing_module("selby")
    powers = np.array(
        [
            [1.0, 3.0],
            [2.0, 4.0],
            [10.0, 5.0],
            [4.0, 7.0],
        ]
    )
    noise = selby.rolling_noise_floor(powers, noise_window_len=2)
    expected = np.array(
        [
            [1.0, 3.0],
            [1.5, 3.5],
            [6.0, 4.5],
            [7.0, 6.0],
        ]
    )
    np.testing.assert_allclose(noise, expected)


def test_compute_selby_score_cube_tracks_coherent_arrival(load_processing_module):
    selby = load_processing_module("selby")
    fs = 40.0
    duration_s = 8.0
    centre_freq_hz = 5.0
    base_signal = _make_burst_signal(fs, duration_s, centre_freq_hz)

    xx = np.array([-0.5, 0.5, -0.5, 0.5])
    yy = np.array([-0.5, -0.5, 0.5, 0.5])
    true_slow = 0.2
    true_bazi = 90.0

    data = []
    rng = np.random.default_rng(12)
    for x_rec, y_rec in zip(xx, yy):
        dt_s = x_rec * true_slow * np.sin(np.deg2rad(true_bazi)) + y_rec * true_slow * np.cos(
            np.deg2rad(true_bazi)
        )
        trace = _shift_signal(base_signal, dt_s, fs)
        trace += 0.03 * rng.standard_normal(len(trace))
        data.append(trace)
    data = np.asarray(data)

    score_cube, centres = selby.compute_selby_score_cube(
        data,
        fs,
        xx,
        yy,
        min_sl=0.0,
        max_sl=0.3,
        n_sl=16,
        min_baz=0.0,
        max_baz=180.0,
        n_baz=19,
        win_len_s=1.0,
        win_step_inc_s=0.5,
        freqmin=4.0,
        freqmax=6.0,
        filter_step_hz=0.5,
        filter_half_width_hz=0.5,
        noise_window_s=1.5,
    )

    assert score_cube.shape[0] == len(centres)
    slow_grid = np.linspace(0.0, 0.3, 16)
    baz_grid = np.linspace(0.0, 180.0, 19)
    slow_idx = np.argmin(np.abs(slow_grid - true_slow))
    baz_idx = np.argmin(np.abs(baz_grid - true_bazi))
    true_track = score_cube[:, slow_idx, baz_idx]

    assert np.max(true_track) > np.median(score_cube)
    assert np.max(true_track) > np.percentile(score_cube, 90)


def test_compute_selby_score_cube_prefers_signal_over_noise(load_processing_module):
    selby = load_processing_module("selby")
    fs = 40.0
    xx = np.array([-0.1, 0.1, -0.1, 0.1])
    yy = np.array([-0.1, -0.1, 0.1, 0.1])
    rng = np.random.default_rng(7)

    noise_only = 0.05 * rng.standard_normal((4, int(8.0 * fs)))
    signal = noise_only.copy()
    signal += np.vstack([_make_burst_signal(fs, 8.0, 4.5)] * 4)

    noise_cube, _ = selby.compute_selby_score_cube(
        noise_only,
        fs,
        xx,
        yy,
        min_sl=0.0,
        max_sl=0.2,
        n_sl=9,
        min_baz=0.0,
        max_baz=180.0,
        n_baz=13,
        win_len_s=1.0,
        win_step_inc_s=0.5,
        freqmin=3.5,
        freqmax=5.5,
        filter_step_hz=0.5,
        filter_half_width_hz=0.5,
        noise_window_s=1.5,
    )
    signal_cube, _ = selby.compute_selby_score_cube(
        signal,
        fs,
        xx,
        yy,
        min_sl=0.0,
        max_sl=0.2,
        n_sl=9,
        min_baz=0.0,
        max_baz=180.0,
        n_baz=13,
        win_len_s=1.0,
        win_step_inc_s=0.5,
        freqmin=3.5,
        freqmax=5.5,
        filter_step_hz=0.5,
        filter_half_width_hz=0.5,
        noise_window_s=1.5,
    )

    assert np.max(signal_cube) > np.max(noise_cube)


def test_compute_selby_score_cube_returns_empty_for_too_short_input(load_processing_module):
    selby = load_processing_module("selby")
    fs = 20.0
    data = np.ones((3, 10))
    xx = np.array([0.0, 0.1, -0.1])
    yy = np.array([0.0, -0.1, 0.1])

    score_cube, centres = selby.compute_selby_score_cube(
        data,
        fs,
        xx,
        yy,
        min_sl=0.0,
        max_sl=0.2,
        n_sl=5,
        min_baz=0.0,
        max_baz=180.0,
        n_baz=7,
        win_len_s=1.0,
        win_step_inc_s=0.5,
        freqmin=1.0,
        freqmax=5.0,
        filter_step_hz=0.5,
        filter_half_width_hz=0.5,
        noise_window_s=1.0,
    )

    assert score_cube.shape == (0, 5, 7)
    assert centres.size == 0


def test_rolling_noise_floor_rejects_non_positive_window_length(load_processing_module):
    selby = load_processing_module("selby")
    powers = np.ones((3, 2))

    try:
        selby.rolling_noise_floor(powers, noise_window_len=0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:  # pragma: no cover - defensive failure message
        raise AssertionError("rolling_noise_floor should reject non-positive windows.")
