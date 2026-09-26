import json

import pandas as pd
import pytest

from pinglab.session import load_session


def test_client_never_falls_back_to_demo(tmp_path):
    config = dict(source_kind="client", manifest="missing.csv", annotations="review.json", output="out")
    (tmp_path / "project.json").write_text(json.dumps(config))
    with pytest.raises(FileNotFoundError):
        load_session(tmp_path)
    assert not (tmp_path / "data/demo").exists()


def test_client_bandwidth_and_metadata(tmp_path):
    import numpy as np
    import soundfile as sf

    sf.write(tmp_path / "a.wav", np.random.default_rng(1).normal(0, 0.01, 16000), 16000)
    pd.DataFrame(
        [
            dict(
                path="a.wav",
                plate_id="one",
                label="unknown",
                session_id="one",
                plate_type="push",
                temperature="25",
            )
        ]
    ).to_csv(tmp_path / "manifest.csv", index=False)
    (tmp_path / "project.json").write_text(
        json.dumps(
            dict(
                source_kind="client",
                manifest="manifest.csv",
                annotations="review.json",
                output="out",
                fmax=20000,
            )
        )
    )
    session = load_session(tmp_path)
    assert session.fmax == 8000 and session.source_kind == "client"
    assert session.records.plate_type.iloc[0] == "push"
    assert session.pair()[0].plate_id == "one"
