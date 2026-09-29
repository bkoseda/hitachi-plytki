import numpy as np
import pandas as pd

from pinglab.clientexplanations import file_block_permutation, mfcc_background, quiet_windows


def test_permutation_moves_whole_recordings_without_mixing_taps():
    frame = pd.DataFrame({"recording_id": np.repeat(["a", "b", "c"], 10)})
    index = file_block_permutation(frame, np.random.default_rng(11))
    assert sorted(index) == list(range(30))
    for start in [0, 10, 20]:
        donor = index[start : start + 10]
        assert len(set(frame.iloc[donor].recording_id)) == 1
        np.testing.assert_array_equal(np.diff(donor), np.ones(9))


def test_quiet_windows_exclude_candidate_attack_and_tail():
    sr = 48000
    x = np.ones(sr * 3) * 0.01
    onsets = np.array([0.3, 1.0, 2.1])
    windows = quiet_windows(x, sr, onsets, max_windows=100)
    assert windows
    for _, start, end in windows:
        assert not np.any((start < onsets + 0.4) & (end > onsets - 0.1))
        assert abs(end - start - 0.1) < 1e-9


def test_background_silence_is_finite_and_no_absolute_gain_feature():
    sr = 48000
    silent = mfcc_background(np.zeros(4800), sr)
    assert len(silent) == 26 and all(v == 0 for v in silent.values())
    noise = np.random.default_rng(42).normal(size=4800) * 0.01
    a, b = mfcc_background(noise, sr), mfcc_background(noise * 0.1, sr)
    np.testing.assert_allclose(list(a.values()), list(b.values()), atol=1e-10)
