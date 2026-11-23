import pandas as pd

from Agents.data_repair_service import analyse_repairs, apply_confirmed_repairs


def source_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "材料号": " P-001 ",
                "厚度": 1.2,
                "宽度": 1000.0,
                "外径": 1500.0,
                "毛重": 20.0,
                "净重": 19.8,
                "到站": "Sydny",
                "收货单位": None,
                "备注": "  Priority   delivery  ",
            }
        ]
    )


def mapping() -> dict[str, str]:
    return {
        "material_no": "材料号",
        "thickness": "厚度",
        "width": "宽度",
        "outer_diameter": "外径",
        "gross_weight_tonne": "毛重",
        "net_weight_tonne": "净重",
        "destination": "到站",
        "receiver": "收货单位",
        "remarks": "备注",
    }


def test_exact_material_master_data_can_fill_and_correct_with_confirmation() -> None:
    source = source_dataframe()
    original = source.copy(deep=True)
    master = pd.DataFrame(
        [
            {
                "material_no": "P-001",
                "destination": "Sydney",
                "receiver": "BAJC Sydney DC",
            }
        ]
    )

    analysis = analyse_repairs(
        source,
        mapping(),
        master_dataframe=master,
        master_source_name="approved_master.xlsx",
    )

    by_field = {item["field"]: item for item in analysis.suggestions}
    assert by_field["destination"]["repair_type"] == "authoritative_correction"
    assert by_field["destination"]["changes_loading_input"] is True
    assert by_field["receiver"]["repair_type"] == "authoritative_fill"
    assert by_field["remarks"]["repair_type"] == "deterministic_format_repair"

    selected = [
        by_field["destination"]["repair_id"],
        by_field["receiver"]["repair_id"],
        by_field["remarks"]["repair_id"],
    ]
    repaired, repaired_mapping, changes = apply_confirmed_repairs(
        source, mapping(), analysis.suggestions, selected
    )

    assert source.equals(original)
    assert repaired.iloc[0][repaired_mapping["destination"]] == "Sydney"
    assert repaired.iloc[0][repaired_mapping["receiver"]] == "BAJC Sydney DC"
    assert repaired.iloc[0][repaired_mapping["remarks"]] == "Priority delivery"
    assert len(changes) == 3


def test_missing_fact_without_master_evidence_remains_unresolved() -> None:
    analysis = analyse_repairs(source_dataframe(), mapping())

    unresolved_fields = {item["field"] for item in analysis.unresolved}
    assert "receiver" in unresolved_fields
    assert not any(
        item["field"] == "receiver" for item in analysis.suggestions
    )


def test_duplicate_master_material_is_not_used_as_authority() -> None:
    master = pd.DataFrame(
        [
            {"material_no": "P-001", "destination": "Sydney"},
            {"material_no": "P-001", "destination": "Melbourne"},
        ]
    )

    analysis = analyse_repairs(
        source_dataframe(), mapping(), master_dataframe=master
    )

    assert analysis.duplicate_master_materials == ["P-001"]
    assert not any(
        item["repair_type"].startswith("authoritative")
        for item in analysis.suggestions
    )
