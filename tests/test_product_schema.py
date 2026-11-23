from Agents.product_schema import validate_product_records


def _standard_product(**overrides):
    product = {
        "material_no": "P001",
        "thickness": 1.2,
        "width": 1000,
        "outer_diameter": 1200,
        "gross_weight_tonne": 10.0,
        "net_weight_tonne": 9.8,
        "destination": "Sydney",
    }
    product.update(overrides)
    return product


def test_rejects_kilogram_scale_value_in_tonne_field() -> None:
    issues = validate_product_records(
        [_standard_product(gross_weight_tonne=12_400, net_weight_tonne=12_000)]
    )

    assert any("exceeds the deployed single-product limit" in issue for issue in issues)


def test_accepts_weight_within_deployed_single_product_limit() -> None:
    assert validate_product_records([_standard_product(gross_weight_tonne=60.0)]) == []


def test_rejects_excessive_downstream_weight_in_kilograms() -> None:
    issues = validate_product_records(
        [{
            "product_id": "P001",
            "length": 1200,
            "width": 1000,
            "weight": 12_400_000,
            "destination": "Sydney",
        }]
    )

    assert any("kilograms" in issue for issue in issues)
