# Evidence-Driven Data Repair Workflow

## Release boundary

The original order file is immutable. The system creates repair suggestions,
the operator reviews every suggestion, and only selected items are applied to a
separate planning working copy. An unselected suggestion retains the original
value. A missing field without authoritative evidence remains blank.

## Accepted evidence

1. Deterministic formatting evidence from the original cell, such as leading,
   trailing, or repeated whitespace.
2. An exact `material_no` match in an operator-supplied authoritative master
   file. Fuzzy material matching is not allowed.
3. Deterministically derived downstream fields already defined by the canonical
   product schema.

Duplicate material numbers in the master file are treated as ambiguous and are
not used. Core loading fields can be corrected only through an exact master-data
match and explicit operator confirmation before planning.

## Master-data schema

Supported formats are `.xlsx`, `.xls`, `.csv`, and `.json`. The file must
contain `material_no` and at least one other supported field:

```text
material_no, thickness, width, outer_diameter,
gross_weight_tonne, net_weight_tonne, destination,
product_name, receiver, contract_no, storage_location, remarks
```

Chinese business aliases are also accepted. A header-only template is provided
in `docs/AUTHORITATIVE_MASTER_DATA_TEMPLATE.csv` and in the business UI.

## Persisted evidence

Each completed run may contain:

- `original_input.<suffix>` — byte-preserved operator upload;
- `authoritative_master.<suffix>` — byte-preserved evidence source;
- `repaired_source.xlsx` — original column layout with confirmed repairs;
- `repaired_products.xlsx` — canonical product workbook used for hand-off;
- `data_repair_audit.json` — suggestions, decisions, identities, timestamps,
  hashes, unresolved fields, and immutable-source declaration;
- `products.json` — canonical planning input.

All persisted repair artifacts are hashed in `run_manifest.json` and verified
before the business UI displays or downloads them.
