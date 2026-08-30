import json
import os
import tempfile
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class ResultsStore:
    """Atomically merge one algorithm result into a dataset result file."""

    def __init__(self, output_dir: str | Path):
        output_dir = Path(output_dir)
        self.output_dir = (
            output_dir if output_dir.is_absolute() else REPOSITORY_ROOT / output_dir
        )

    def update(self, dataset: str, algorithm: str, analysis: dict[str, Any]) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"results_{dataset}.json"
        document = self._read(output_path)
        document["results"][algorithm] = analysis

        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.output_dir,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as output_file:
                temporary_path = Path(output_file.name)
                json.dump(
                    document,
                    output_file,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                output_file.write("\n")
                output_file.flush()
                os.fsync(output_file.fileno())

            os.replace(temporary_path, output_path)
        except Exception:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
            raise

        return output_path

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {"results": {}}

        with path.open(encoding="utf-8") as input_file:
            document = json.load(input_file)

        if not isinstance(document, dict) or not isinstance(
            document.get("results"), dict
        ):
            raise ValueError(f"Invalid results document: {path}")

        return document
