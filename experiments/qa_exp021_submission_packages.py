"""Independent isolated and row-independence QA for final EXP-021 ZIPs."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
TARGET = "control_success"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_case(stage: Path, test: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    data = stage / "data"
    output = stage / "output"
    data.mkdir(exist_ok=True)
    if output.exists():
        for path in output.iterdir():
            path.unlink()
    test.to_csv(data / "test.csv", index=False, encoding="utf-8-sig")
    sample.to_csv(
        data / "sample_submission.csv", index=False, encoding="utf-8-sig"
    )
    subprocess.run(
        [str(PYTHON), "script.py"],
        cwd=stage,
        check=True,
        capture_output=True,
        text=True,
    )
    return pd.read_csv(output / "submission.csv")


def audit(archive_path: Path) -> dict[str, object]:
    test = pd.read_csv(ROOT / "data" / "test.csv", encoding="utf-8-sig")
    sample = pd.read_csv(
        ROOT / "data" / "sample_submission.csv", encoding="utf-8-sig"
    )
    with tempfile.TemporaryDirectory(prefix="exp021-independent-qa-") as raw:
        stage = Path(raw)
        with zipfile.ZipFile(archive_path) as archive:
            if archive.testzip() is not None:
                raise AssertionError("CRC failure")
            archive.extractall(stage)
        full = run_case(stage, test, sample)
        repeated = run_case(stage, test, sample)
        if not np.array_equal(
            full[TARGET].to_numpy(), repeated[TARGET].to_numpy()
        ):
            raise AssertionError("repeated inference is not deterministic")

        permutation = np.arange(len(test))[::-1]
        reversed_output = run_case(
            stage,
            test.iloc[permutation].reset_index(drop=True),
            sample.iloc[permutation].reset_index(drop=True),
        )
        expected = full.set_index("row_id")[TARGET]
        actual_reversed = reversed_output.set_index("row_id")[TARGET]
        if not np.array_equal(
            expected.sort_index().to_numpy(),
            actual_reversed.sort_index().to_numpy(),
        ):
            raise AssertionError("row permutation changed predictions")

        singleton_values: dict[str, float] = {}
        for index in range(len(test)):
            one = run_case(
                stage,
                test.iloc[[index]].reset_index(drop=True),
                sample.loc[
                    sample["row_id"].eq(test.iloc[index]["row_id"])
                ].reset_index(drop=True),
            )
            singleton_values[str(one.iloc[0]["row_id"])] = float(
                one.iloc[0][TARGET]
            )
        singleton = pd.Series(singleton_values).sort_index()
        if not np.array_equal(
            expected.sort_index().to_numpy(), singleton.to_numpy()
        ):
            raise AssertionError("singleton predictions differ from batch")

        model = stage / "model"
        metadata = json.loads(
            (model / "metadata.json").read_text(encoding="utf-8")
        )
        return {
            "zip": str(archive_path),
            "sha256": sha256(archive_path),
            "candidate": metadata["candidate"],
            "rows": int(len(full)),
            "train_csv_absent": not (stage / "data" / "train.csv").exists(),
            "repeat_exact": True,
            "permutation_exact": True,
            "singleton_exact": True,
            "row_id_order": bool(full["row_id"].equals(sample["row_id"])),
            "finite_0_1": bool(
                np.isfinite(full[TARGET]).all()
                and full[TARGET].between(0.0, 1.0).all()
            ),
        }


def main() -> None:
    results = [
        audit(ROOT / "submit_exp021_strict.zip"),
        audit(ROOT / "submit_exp021_aggr.zip"),
    ]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
