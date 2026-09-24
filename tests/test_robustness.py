import numpy as np
import pytest

from pinglab.robustness import mix_noise


def test_requested_snr():
    rng = np.random.default_rng(42)
    y = rng.normal(0, 0.1, 10000)
    noise = rng.normal(size=20000)
    out = mix_noise(y, noise, 10)
    actual = 10 * np.log10(np.mean(y * y) / np.mean((out - y) ** 2))
    assert abs(actual - 10) < 0.01
    with pytest.raises(ValueError):
        mix_noise(y, np.zeros(100), 10)
