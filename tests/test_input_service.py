from types import SimpleNamespace

import pytest

from Agents.input_service import save_uploaded_products


def test_critical_weight_conflict_never_enters_llm_fallback(
    tmp_path, monkeypatch
) -> None:
    products = [{
        "material_no": "P001",
        "thickness": 1.2,
        "width": 1000,
        "outer_diameter": 1200,
        "gross_weight_tonne": 12_400,
        "net_weight_tonne": 12_000,
        "destination": "Sydney",
    }]
    monkeypatch.setattr(
        "Agents.input_service.DataAgent.run",
        lambda *args, **kwargs: products,
    )

    def unexpected_llm_call(*args, **kwargs):
        raise AssertionError("LLM field mapping must not run for a critical weight conflict")

    monkeypatch.setattr(
        "Agents.input_service.infer_product_field_mapping",
        unexpected_llm_call,
    )
    upload = SimpleNamespace(name="wrong_weight_unit.xlsx", getbuffer=lambda: b"fixture")

    with pytest.raises(ValueError, match="LLM assistance is not permitted"):
        save_uploaded_products(
            upload,
            tmp_path,
            llm_data_recovery_mode="Agent decides",
            llm_model_name="unused",
            llm_base_url="http://unused",
        )
