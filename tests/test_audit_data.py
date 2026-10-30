import pandas as pd
import pytest

from uasd_ds_2_final_project.audit import prepare_dataset


def test_prepare_dataset_encodes_target_and_preserves_rows_and_unknowns():
    source = pd.DataFrame(
        {
            "job": [" admin. ", "unknown"],
            "y": ["yes", " no "],
        }
    )

    prepared = prepare_dataset(source)

    assert len(prepared) == len(source)
    assert prepared["y"].tolist() == [1, 0]
    assert prepared["job"].tolist() == ["admin.", "unknown"]


def test_prepare_dataset_rejects_unexpected_target_labels():
    with pytest.raises(ValueError, match="Etiquetas no válidas"):
        prepare_dataset(pd.DataFrame({"y": ["maybe"]}))
