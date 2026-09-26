import numpy as np
import pytest
import soundfile as sf

from pinglab.recorder import demo_sequence, export_pings, record_wav, segment_ringdown


def test_complete_and_overlapping_decays_detected():
    y, sr, truth = demo_sequence()
    table, t, db = segment_ringdown(y, sr)
    np.testing.assert_allclose(table.onset_s, truth, atol=0.025)
    assert table.status.tolist() == [
        "zanik zaobserwowany",
        "kolejne uderzenie przed zanikiem",
        "zanik zaobserwowany",
        "zanik zaobserwowany",
    ]
    assert table.iloc[1].end_s == table.iloc[2].onset_s
    assert table.iloc[0].end_s < table.iloc[1].onset_s
    short, *_ = segment_ringdown(y[: int(6.2 * sr)], sr, onsets=truth)
    assert short.iloc[-1].status == "koniec nagrania przed zanikiem"


def test_silence_and_invalid_onsets():
    table, *_ = segment_ringdown(np.zeros(48000), 48000)
    assert table.empty
    y, sr, truth = demo_sequence()
    with pytest.raises(ValueError, match="rosnące"):
        segment_ringdown(y, sr, onsets=[2, 1])
    table, *_ = segment_ringdown(y, sr, onsets=[])
    assert table.empty


def test_recording_native_rate_unique_files_and_overflow(monkeypatch, tmp_path):
    # No microphone is activated by unit tests.
    import sounddevice as sd

    args = []
    monkeypatch.setattr(sd, "query_devices", lambda *a: dict(default_samplerate=44100))
    monkeypatch.setattr(sd, "check_input_settings", lambda **k: None)

    def rec(n, **kwargs):
        args.append(kwargs)
        return np.zeros((n, 1), np.float32)

    monkeypatch.setattr(sd, "rec", rec)
    monkeypatch.setattr(sd, "wait", lambda: None)
    a = record_wav(tmp_path, seconds=1, countdown=0)
    b = record_wav(tmp_path, seconds=1, countdown=0)
    assert a != b and sf.info(a).samplerate == 44100 and args[0]["channels"] == 1
    monkeypatch.setattr(sd, "wait", lambda: "input overflow")
    with pytest.raises(RuntimeError, match="Przerwa"):
        record_wav(tmp_path, seconds=1, countdown=0)
    assert len(list(tmp_path.glob("*.wav"))) == 2


def test_segments_stop_before_next_hit(tmp_path):
    y, sr, truth = demo_sequence()
    table, *_ = segment_ringdown(y, sr, onsets=truth)
    paths = export_pings(y, sr, table, tmp_path)
    assert len(paths) == 4
    info = sf.info(paths[1])
    assert info.duration == pytest.approx(0.365, abs=1 / sr)
    with pytest.raises(FileExistsError):
        export_pings(y, sr, table, tmp_path)
