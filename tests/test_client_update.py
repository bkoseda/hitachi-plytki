import pandas as pd
import pytest

from pinglab.clientmodels import group_folds
from pinglab.clientupdate import SCENARIOS, inventory, scenario_data


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_ambiguous_plate_aliases_never_cross_folds(scenario):
    names = ["Bad_1", "Bad_2", "NOK_LAB_BAD2", "OK_1", "OK_2", "OK_3", "OK_ENH_OFF", "OK_LAB"]
    frame = pd.DataFrame(
        [
            dict(group=g, label=int(i < 3), scope="core", recording_id=f"r{i}", event_id=f"e{i}")
            for i, g in enumerate(names)
        ]
    )
    d = scenario_data(frame, scenario)
    # All possibly identical new OK recordings are kept with assumed OK_1.
    assert d[d.recording_id.isin(["r3", "r6", "r7"])].group.nunique() == 1
    for _, tr, te in group_folds(d):
        assert set(d.iloc[tr].recording_id).isdisjoint(d.iloc[te].recording_id)
        assert set(d.iloc[tr].group).isdisjoint(d.iloc[te].group)
    if scenario == "same_BAD2":
        assert d[d.recording_id.isin(["r1", "r2"])].group.nunique() == 1
    if scenario == "new_BAD2_merge_old":
        assert d[d.recording_id.isin(["r0", "r1"])].group.nunique() == 1


def test_names_do_not_invent_processing_or_support():
    data = inventory()
    assert len(data) == 16
    lab = data[data.recording_id.str.startswith("L")]
    assert lab.processing.eq("unknown_LAB_TEST").all()
    assert lab.support.eq("unknown").all()
    assert lab[lab.label.eq(1)].group.nunique() == 1
    assert data[data.scope.eq("background")].label.eq(-1).all()
