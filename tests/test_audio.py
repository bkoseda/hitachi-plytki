import numpy as np
import pytest
import soundfile as sf

from pinglab.audio import Config, cut_hits, detect_hits, load_audio
from pinglab.demo import resonator
from pinglab.features import extract_features


def test_silence_no_invented_hit(tmp_path):
    y = np.zeros(48000)
    path = tmp_path / "silence.wav"
    sf.write(path, y, 48000)
    a = load_audio(path)
    assert len(detect_hits(a.y, a.sr)) == 0
    assert any("CISZA" in w for w in a.qc["warnings"])
    with pytest.raises(ValueError, match="Cisza"):
        extract_features(y, 48000)


def test_three_hits_and_no_crossing():
    sr = 48000
    y = np.zeros(sr * 4)
    for t in [0.3, 1.5, 2.7]:
        hit = resonator(sr, 0.9)
        y[round(t * sr) : round(t * sr) + len(hit)] += hit
    times = detect_hits(y, sr)
    assert len(times) == 3
    assert np.allclose(times, [0.3, 1.5, 2.7], atol=0.035)
    clips = cut_hits(y, sr, times, Config(tail_s=2))
    for a, b in zip(clips, clips[1:]):
        assert a["end_s"] <= b["start_s"]
    assert clips[0]["truncated"]


def test_start_hit():
    y = resonator()
    times = detect_hits(y, 48000)
    assert len(times) == 1 and times[0] < 0.03


def test_stereo_antiphase_preserved(tmp_path):
    y = resonator()
    path = tmp_path / "stereo.wav"
    sf.write(path, np.stack([y, -y], axis=1), 48000)
    a = load_audio(path)
    assert a.channels == 2 and np.max(np.abs(a.y)) > 0.9
    assert len(detect_hits(a.y, a.sr)) == 1
    with pytest.raises(ValueError, match="kanału"):
        load_audio(path, channel=3)


def test_clipping_original_measured(tmp_path):
    y = np.ones(48000) * 1.2
    path = tmp_path / "clip.wav"
    sf.write(path, y, 48000, subtype="FLOAT")
    assert load_audio(path).qc["clipping_fraction"] == 1


def test_bad_manual_times():
    for times in [[0.5, 0.4], [0.5, 0.5], [-1], [2], [np.nan], [0.4, 0.401]]:
        with pytest.raises(ValueError):
            cut_hits(np.ones(48000), 48000, times)


def test_gain_invariance():
    y = np.r_[np.zeros(1200), resonator()]
    a = extract_features(y, 48000)
    b = extract_features(y * 0.1, 48000)
    np.testing.assert_allclose(list(a.values()), list(b.values()), rtol=1e-5, atol=1e-5)


def test_nan_empty_and_duration(tmp_path):
    for name, y in [("nan", np.full(48000, np.nan)), ("tiny", np.zeros(20))]:
        path = tmp_path / f"{name}.wav"
        sf.write(path, y, 48000, subtype="FLOAT")
        with pytest.raises(ValueError):
            load_audio(path)
    path = tmp_path / "long.wav"
    sf.write(path, np.zeros(48000), 48000)
    with pytest.raises(ValueError, match="Nagranie"):
        load_audio(path, Config(max_seconds=0.5))


@pytest.mark.parametrize(
    "suffix,codec", [(".mp3", "libmp3lame"), (".m4a", "aac"), (".flac", "flac"), (".ogg", "libvorbis")]
)
def test_codecs(tmp_path, suffix, codec):
    import subprocess

    import imageio_ffmpeg

    src = tmp_path / "source.wav"
    dst = tmp_path / f"encoded{suffix}"
    sf.write(src, resonator(), 48000)
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(src), "-c:a", codec, str(dst)], check=True
    )
    a = load_audio(dst)
    assert a.sr == 48000 and a.qc["duration_s"] > 1
    assert len(detect_hits(a.y, a.sr)) >= 1
