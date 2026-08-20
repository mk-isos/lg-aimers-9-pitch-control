"""Independent source-only QA for the exploratory post-EXP-058 ZIPs.

The canonical competition test and sample-submission files are never opened.
Inference fixtures are derived from six training rows, have their target
removed, their season changed to 2025, and their raw identifiers changed to
unseen values so that frozen-history consistency checks remain meaningful.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
READY_DIR = ROOT / "ready_to_submit" / "2026-08-20-post58"
TRAIN_PATH = ROOT / "data" / "train.csv"
PYTHON = ROOT / ".venv" / "bin" / "python"
ID_COL = "row_id"
TARGET_COL = "control_success"
TOLERANCE = 1e-9
RAW_ID_COLUMNS = (
    "pitcher_id",
    "batter_id",
    "pitcher_team_id",
    "batter_team_id",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixture() -> pd.DataFrame:
    frame = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig", nrows=6)
    frame = frame.drop(columns=[TARGET_COL])
    frame["season"] = 2025
    for offset, column in enumerate(RAW_ID_COLUMNS, start=1):
        frame[column] = np.arange(len(frame), dtype=np.int64) + offset * 1_000_000_000
    # The synthetic identifiers are absent from every frozen 2019-2024 state,
    # so their official career counters must describe the same zero-history
    # condition.  This exercises the package's unseen-entity path without
    # weakening its history-consistency checks.
    for column in (
        "asof_pitcher_n",
        "asof_batter_n",
        "asof_pitcher_pitchmix_n",
    ):
        frame[column] = 0
    frame[ID_COL] = [f"SOURCE_QA_{index:03d}" for index in range(len(frame))]
    return frame


def run(stage: Path, rows: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    data_dir = stage / "data"
    output_dir = stage / "output"
    data_dir.mkdir(exist_ok=True)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    rows.to_csv(data_dir / "test.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {ID_COL: rows[ID_COL], TARGET_COL: np.full(len(rows), 0.5)}
    ).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
        encoding="utf-8-sig",
    )
    started = time.time()
    completed = subprocess.run(
        [str(PYTHON), "script.py"],
        cwd=stage,
        check=True,
        capture_output=True,
        text=True,
    )
    runtime = time.time() - started
    prediction = pd.read_csv(output_dir / "submission.csv", encoding="utf-8-sig")
    if prediction.columns.tolist() != [ID_COL, TARGET_COL]:
        raise ValueError("submission schema mismatch")
    if prediction[ID_COL].tolist() != rows[ID_COL].tolist():
        raise ValueError("submission order mismatch")
    values = prediction[TARGET_COL].to_numpy(float)
    if not np.isfinite(values).all() or not ((values >= 0.0) & (values <= 1.0)).all():
        raise ValueError("invalid submission probabilities")
    if not completed.stdout.strip():
        raise ValueError("submission script emitted no completion message")
    return prediction, runtime


def aligned_values(prediction: pd.DataFrame, identifiers: list[str]) -> np.ndarray:
    return (
        prediction.set_index(ID_COL)[TARGET_COL]
        .astype(float)
        .reindex(identifiers)
        .to_numpy(float)
    )


def audit_zip(path: Path, rows: pd.DataFrame) -> dict[str, object]:
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"CRC failure in {path.name}: {bad}")
        names = archive.namelist()
        if names[:2] != ["script.py", "requirements.txt"]:
            raise ValueError(f"root order mismatch in {path.name}")
        if not all(
            name in {"script.py", "requirements.txt"} or name.startswith("model/")
            for name in names
        ):
            raise ValueError(f"unexpected ZIP member in {path.name}")
        forbidden_suffixes = {".csv", ".npy", ".npz", ".pkl", ".pickle", ".joblib"}
        if any(Path(name).suffix.lower() in forbidden_suffixes for name in names):
            raise ValueError(f"training/prediction artifact in {path.name}")
        script = archive.read("script.py").decode("utf-8")
        requirements = archive.read("requirements.txt").decode("utf-8")
    forbidden_script_fragments = (
        "predictions.mean()",
        "predictions.min()",
        "predictions.max()",
        "rows={len(sample)}",
        "requests.",
        "urllib.",
        "http://",
        "https://",
    )
    observed_forbidden = [value for value in forbidden_script_fragments if value in script]
    if observed_forbidden:
        raise ValueError(f"forbidden script fragments in {path.name}: {observed_forbidden}")
    if requirements.strip() != "lightgbm==4.6.0":
        raise ValueError(f"unexpected requirements in {path.name}")

    with tempfile.TemporaryDirectory(prefix="post58-source-qa-") as temporary:
        stage = Path(temporary)
        with zipfile.ZipFile(path) as archive:
            archive.extractall(stage)
        subprocess.run(
            [str(PYTHON), "-m", "py_compile", "script.py"],
            cwd=stage,
            check=True,
            capture_output=True,
            text=True,
        )
        full, full_runtime = run(stage, rows)
        identifiers = rows[ID_COL].tolist()
        reference = aligned_values(full, identifiers)

        reverse_rows = rows.iloc[::-1].reset_index(drop=True)
        reverse, reverse_runtime = run(stage, reverse_rows)
        comparisons = [
            float(np.max(np.abs(reference - aligned_values(reverse, identifiers))))
        ]

        first, first_runtime = run(stage, rows.iloc[:3].reset_index(drop=True))
        second, second_runtime = run(stage, rows.iloc[3:].reset_index(drop=True))
        split = pd.concat([first, second], ignore_index=True)
        comparisons.append(
            float(np.max(np.abs(reference - aligned_values(split, identifiers))))
        )

        singleton_runtime = 0.0
        singleton_predictions: list[pd.DataFrame] = []
        for position in range(len(rows)):
            prediction, runtime = run(stage, rows.iloc[[position]].reset_index(drop=True))
            singleton_predictions.append(prediction)
            singleton_runtime += runtime
        singleton = pd.concat(singleton_predictions, ignore_index=True)
        comparisons.append(
            float(np.max(np.abs(reference - aligned_values(singleton, identifiers))))
        )

        duplicate_rows = rows.iloc[[0, 1, 0]].reset_index(drop=True).copy()
        duplicate_rows.loc[2, ID_COL] = "SOURCE_QA_DUPLICATE"
        duplicate, duplicate_runtime = run(stage, duplicate_rows)
        comparisons.append(
            abs(
                float(duplicate.iloc[0][TARGET_COL])
                - float(duplicate.iloc[2][TARGET_COL])
            )
        )

    maximum = float(max(comparisons))
    if maximum > TOLERANCE:
        raise ValueError(f"row-independence failure in {path.name}: {maximum}")
    return {
        "zip": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "crc": "passed",
        "member_count": len(names),
        "requirements": requirements.strip(),
        "canonical_test_or_sample_opened": False,
        "fixture": "six train rows, target removed, season=2025, unseen raw IDs",
        "invariance": {
            "full_reverse_split_singleton_duplicate": "passed",
            "max_abs_difference": maximum,
            "tolerance": TOLERANCE,
        },
        "runtime_seconds": {
            "full": full_runtime,
            "reverse": reverse_runtime,
            "split_total": first_runtime + second_runtime,
            "singletons_total": singleton_runtime,
            "duplicate": duplicate_runtime,
        },
    }


def main() -> None:
    paths = sorted(READY_DIR.glob("*.zip"))
    if len(paths) != 5:
        raise ValueError(f"expected five post-058 ZIPs, found {len(paths)}")
    rows = fixture()
    audits = [audit_zip(path, rows) for path in paths]
    report = {
        "audit": "post-EXP-058 exploratory submission packages",
        "canonical_test_or_sample_opened": False,
        "packages": audits,
    }
    output = READY_DIR / "qa_report.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"passed={len(audits)} report={output}", flush=True)


if __name__ == "__main__":
    main()
