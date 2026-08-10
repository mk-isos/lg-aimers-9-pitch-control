"""EXP-018 최종 ZIP 하나를 만들고 격리 추론·구조·CRC를 검사한다."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submissions" / "EXP-018"
OUTPUT = ROOT / "submit_exp018_constrained_multiscale.zip"
PYTHON = ROOT / ".venv" / "bin" / "python"
REQUIRED_MODEL_FILES = {
    "encoded_features.json",
    "group_effects.json",
    "history_state.json",
    "metadata.json",
    "recent_residual_lightgbm.txt",
}
FORBIDDEN_SUFFIXES = {
    ".csv",
    ".npy",
    ".npz",
    ".pkl",
    ".pickle",
    ".joblib",
    ".log",
}


def validate_source() -> None:
    for filename in ("script.py", "requirements.txt"):
        if not (SOURCE / filename).is_file():
            raise FileNotFoundError(SOURCE / filename)
    model_dir = SOURCE / "model"
    present = {path.name for path in model_dir.iterdir() if path.is_file()}
    if present != REQUIRED_MODEL_FILES:
        raise ValueError(
            f"모델 파일 불일치: missing={REQUIRED_MODEL_FILES - present}, "
            f"extra={present - REQUIRED_MODEL_FILES}"
        )
    metadata = json.loads(
        (model_dir / "metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("experiment") != "EXP-018":
        raise ValueError("metadata experiment가 EXP-018이 아닙니다.")
    if metadata.get("probability_calibration") != "identity":
        raise ValueError("검증되지 않은 확률 보정이 metadata에 있습니다.")


def build_zip() -> None:
    if OUTPUT.exists():
        OUTPUT.unlink()
    with zipfile.ZipFile(
        OUTPUT,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        archive.write(SOURCE / "script.py", "script.py")
        archive.write(SOURCE / "requirements.txt", "requirements.txt")
        for path in sorted((SOURCE / "model").iterdir()):
            archive.write(path, f"model/{path.name}")


def inspect_zip() -> dict[str, object]:
    with zipfile.ZipFile(OUTPUT) as archive:
        bad_crc = archive.testzip()
        if bad_crc is not None:
            raise ValueError(f"ZIP CRC 실패: {bad_crc}")
        names = archive.namelist()
        if names[:2] != ["script.py", "requirements.txt"]:
            raise ValueError(f"최상위 파일 순서가 잘못됐습니다: {names[:2]}")
        if not all(
            name in {"script.py", "requirements.txt"}
            or name.startswith("model/")
            for name in names
        ):
            raise ValueError(f"허용되지 않은 ZIP 경로가 있습니다: {names}")
        if any(
            Path(name).suffix.lower() in FORBIDDEN_SUFFIXES
            for name in names
        ):
            raise ValueError("ZIP에 데이터·예측·pickle·로그 파일이 포함됐습니다.")
        if any("__MACOSX" in name or name.endswith(".DS_Store") for name in names):
            raise ValueError("ZIP에 macOS 메타데이터가 포함됐습니다.")
        expected = {
            "script.py",
            "requirements.txt",
            *(f"model/{name}" for name in REQUIRED_MODEL_FILES),
        }
        if set(names) != expected:
            raise ValueError(f"ZIP 구조 불일치: {set(names) ^ expected}")
        return {
            "crc": "passed",
            "files": [
                {
                    "name": info.filename,
                    "compressed_bytes": info.compress_size,
                    "raw_bytes": info.file_size,
                }
                for info in archive.infolist()
            ],
        }


def smoke_test() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="exp018-smoke-") as temporary:
        stage = Path(temporary)
        with zipfile.ZipFile(OUTPUT) as archive:
            archive.extractall(stage)
        (stage / "data").symlink_to(ROOT / "data", target_is_directory=True)
        started_at = time.time()
        result = subprocess.run(
            [str(PYTHON), "script.py"],
            cwd=stage,
            check=True,
            capture_output=True,
            text=True,
        )
        runtime_seconds = time.time() - started_at
        output_path = stage / "output" / "submission.csv"
        if not output_path.is_file():
            raise FileNotFoundError(output_path)
        test = pd.read_csv(ROOT / "data" / "test.csv", encoding="utf-8-sig")
        sample = pd.read_csv(
            ROOT / "data" / "sample_submission.csv",
            encoding="utf-8-sig",
        )
        submission = pd.read_csv(output_path, encoding="utf-8-sig")
        if list(submission.columns) != ["row_id", "control_success"]:
            raise ValueError("생성된 submission 컬럼이 잘못됐습니다.")
        if len(submission) != len(test):
            raise ValueError("생성된 submission 행 수가 잘못됐습니다.")
        if not submission["row_id"].equals(sample["row_id"]):
            raise ValueError("생성된 submission row_id 순서가 다릅니다.")
        if submission["row_id"].duplicated().any():
            raise ValueError("생성된 submission row_id가 중복됐습니다.")
        if submission["control_success"].isna().any():
            raise ValueError("생성된 submission 예측에 결측값이 있습니다.")
        if not submission["control_success"].between(0.0, 1.0).all():
            raise ValueError("생성된 submission 예측이 범위를 벗어났습니다.")
        return {
            "runtime_seconds": runtime_seconds,
            "stdout": result.stdout.strip(),
            "rows": int(len(submission)),
            "row_id_order": "passed",
            "row_id_unique": "passed",
            "missing_predictions": 0,
            "probability_range": "passed",
            "prediction_mean": float(submission["control_success"].mean()),
            "prediction_min": float(submission["control_success"].min()),
            "prediction_max": float(submission["control_success"].max()),
        }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    validate_source()
    build_zip()
    zip_result = inspect_zip()
    smoke_result = smoke_test()
    result = {
        "zip": str(OUTPUT),
        "zip_bytes": OUTPUT.stat().st_size,
        "sha256": sha256(OUTPUT),
        "zip_validation": zip_result,
        "smoke_test": smoke_result,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
