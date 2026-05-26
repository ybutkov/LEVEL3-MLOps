# src/bike_rental/defs/resources/bike_data.py
from pathlib import Path
import dagster as dg
import pandas as pd


class BikeDataDirResource(dg.ConfigurableResource):
    base_path: str

    def load_csv(self, filename: str) -> pd.DataFrame:
        return pd.read_csv(Path(self.base_path) / filename)
