"""
Data Agent
==========

DataAgent
cleaned xlsx → products.json

Responsibility:
- Input: a cleaned pandas-friendly Excel file (.xlsx)
- Output: a JSON file that downstream agents can use

Important design decision:
- This DataAgent does NOT clean the original raw Excel file.
- Raw Excel cleaning should be handled before this agent.
- This agent only standardises the already-cleaned xlsx into JSON.

Required package:
    pip install pandas openpyxl

Expected cleaned Excel columns:
    material_no, thickness, width, outer_diameter, gross_weight_tonne, net_weight_tonne, destination

Optional columns:
    source_row, sequence, storage_location, contract_no, product_name, receiver

Downstream JSON fields generated:
    product_id      <- material_no
    length          <- outer_diameter
    width           <- width
    weight          <- gross_weight_tonne * 1000
    destination     <- destination
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class ProductRecord:
    """
    Clean product record exported by DataAgent.

    The first group of fields preserves the cleaned business data.
    The second group of fields is prepared for downstream optimisation agents.
    """

    # Original cleaned business fields
    material_no: str
    thickness: float
    width: float
    outer_diameter: float
    gross_weight_tonne: float
    net_weight_tonne: float
    destination: str

    # Optional metadata fields
    source_row: Optional[int] = None
    sequence: Optional[str] = None
    storage_location: Optional[str] = None
    contract_no: Optional[str] = None
    product_name: Optional[str] = None
    receiver: Optional[str] = None
    remarks: Optional[str] = None

    # Downstream-compatible fields
    product_id: Optional[str] = None
    length: Optional[float] = None
    weight: Optional[float] = None
    gross_weight_kg: Optional[float] = None
    net_weight_kg: Optional[float] = None

    def finalise(self) -> "ProductRecord":
        """Generate fields required by downstream agents."""
        self.product_id = self.material_no
        self.length = self.outer_diameter
        self.weight = self.gross_weight_tonne * 1000
        self.gross_weight_kg = self.gross_weight_tonne * 1000
        self.net_weight_kg = self.net_weight_tonne * 1000
        return self


class DataAgent:
    """
    DataAgent converts a cleaned .xlsx file into a standard JSON file.

    It assumes the input Excel has already been cleaned and has one product per row.
    """

    REQUIRED_COLUMNS = [
        "material_no",
        "thickness",
        "width",
        "outer_diameter",
        "gross_weight_tonne",
        "net_weight_tonne",
        "destination",
    ]

    OPTIONAL_COLUMNS = [
        "source_row",
        "sequence",
        "storage_location",
        "contract_no",
        "product_name",
        "receiver",
    ]

    def run(
        self,
        cleaned_excel_path: str | Path,
        json_output_path: str | Path = "products.json",
    ) -> List[Dict[str, Any]]:
        """
        Main DataAgent pipeline.

        Parameters
        ----------
        cleaned_excel_path:
            Path to the already-cleaned .xlsx file.

        json_output_path:
            Path where the standard JSON file will be saved.

        Returns
        -------
        List[Dict[str, Any]]
            Product records that are also written into the JSON file.
        """
        df = self.load_cleaned_excel(cleaned_excel_path)
        records = self.convert_dataframe_to_records(df)
        self.export_json(records, json_output_path)
        return records

    def load_cleaned_excel(self, cleaned_excel_path: str | Path) -> pd.DataFrame:
        """Load the cleaned pandas-friendly Excel file."""
        cleaned_excel_path = Path(cleaned_excel_path)

        if not cleaned_excel_path.exists():
            raise FileNotFoundError(f"Cleaned Excel file not found: {cleaned_excel_path}")

        df = pd.read_excel(cleaned_excel_path)
        df = self._normalise_column_names(df)
        self._validate_required_columns(df)
        return df

    def convert_dataframe_to_records(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert the cleaned dataframe into JSON-ready product records."""
        records: List[Dict[str, Any]] = []

        for index, row in df.iterrows():
            record = ProductRecord(
                material_no=self._as_string(row["material_no"]),
                thickness=self._as_float(row["thickness"]),
                width=self._as_float(row["width"]),
                outer_diameter=self._as_float(row["outer_diameter"]),
                gross_weight_tonne=self._as_float(row["gross_weight_tonne"]),
                net_weight_tonne=self._as_float(row["net_weight_tonne"]),
                destination=self._as_string(row["destination"]),
                source_row=self._as_optional_int(row.get("source_row", index + 2)),
                sequence=self._as_optional_string(row.get("sequence")),
                storage_location=self._as_optional_string(row.get("storage_location")),
                contract_no=self._as_optional_string(row.get("contract_no")),
                product_name=self._as_optional_string(row.get("product_name")),
                receiver=self._as_optional_string(row.get("receiver")),
                remarks=self._as_optional_string(row.get("remarks")),
            ).finalise()

            records.append(asdict(record))

        return records

    def export_json(self, records: List[Dict[str, Any]], json_output_path: str | Path) -> Path:
        """Write product records to JSON."""
        json_output_path = Path(json_output_path)
        json_output_path.parent.mkdir(parents=True, exist_ok=True)

        with json_output_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        return json_output_path

    def _normalise_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise common column name variants.

        This allows the cleaned Excel to use either English names or original Chinese names.
        """
        column_mapping = {
            "材料号": "material_no",
            "厚度": "thickness",
            "宽度": "width",
            "外径": "outer_diameter",
            "毛重": "gross_weight_tonne",
            "净重": "net_weight_tonne",
            "到站": "destination",
            "序号": "sequence",
            "库位": "storage_location",
            "合同号": "contract_no",
            "品名": "product_name",
            "收货用户": "receiver",
            "收货单位": "receiver",
            "备注": "remarks",
        }

        renamed_columns = {}
        for col in df.columns:
            clean_col = str(col).strip()
            renamed_columns[col] = column_mapping.get(clean_col, clean_col)

        return df.rename(columns=renamed_columns)

    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        missing_columns = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_columns:
            raise ValueError(
                "The cleaned Excel file is missing required columns: "
                f"{missing_columns}. Required columns are: {self.REQUIRED_COLUMNS}"
            )

    @staticmethod
    def _as_string(value: Any) -> str:
        if pd.isna(value):
            return ""
        text = str(value).strip()
        if text.endswith(".0") and text[:-2].isdigit():
            text = text[:-2]
        return text

    @staticmethod
    def _as_optional_string(value: Any) -> Optional[str]:
        if value is None or pd.isna(value):
            return None
        text = str(value).strip()
        if not text:
            return None
        if text.endswith(".0") and text[:-2].isdigit():
            text = text[:-2]
        return text

    @staticmethod
    def _as_float(value: Any) -> float:
        if pd.isna(value):
            raise ValueError("Required numeric value is missing.")
        return float(value)

    @staticmethod
    def _as_optional_int(value: Any) -> Optional[int]:
        if value is None or pd.isna(value):
            return None
        return int(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert cleaned Excel product data into JSON.")
    parser.add_argument(
        "--input",
        "-i",
        default="products_clean.xlsx",
        help="Path to the cleaned Excel file. Default: products_clean.xlsx",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="products.json",
        help="Path to the output JSON file. Default: products.json",
    )

    args = parser.parse_args()

    agent = DataAgent()
    records = agent.run(
        cleaned_excel_path=args.input,
        json_output_path=args.output,
    )
    print(f"DataAgent completed. Exported {len(records)} product records to {args.output}")


if __name__ == "__main__":
    main()
