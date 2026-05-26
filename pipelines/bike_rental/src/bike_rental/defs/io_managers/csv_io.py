from pathlib import Path
import dagster as dg
import pandas as pd

class CSVIOManager(dg.IOManager):

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, context) -> Path:
        return self.base_dir / f"{context.asset_key.path[-1]}.csv"

    def handle_output(self, context: dg.OutputContext, obj):
        df = obj.value if isinstance(obj, dg.MaterializeResult) else obj
        df.to_csv(self._path(context), index=False)

    def load_input(self, context: dg.InputContext) -> pd.DataFrame:
        return pd.read_csv(self._path(context))
