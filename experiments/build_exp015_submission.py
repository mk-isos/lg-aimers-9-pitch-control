"""EXP-015 제출 ZIP을 만들고 평가 환경 형태로 샘플 추론한다."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submissions" / "EXP-015"
OUTPUT = ROOT / "submit_exp015_best.zip"
PYTHON = ROOT / ".venv" / "bin" / "python"
REQUIRED_MODEL_FILES = {
    "catboost_model.cbm",
    "engineered_features.json",
    "lightgbm_columns.json",
    "lightgbm_model.txt",
    "metadata.json",
}


def validate_source() -> None:
    for filename in ("script.py", "requirements.txt"):
        if not (SOURCE / filename).is_file():
            raise FileNotFoundError(SOURCE / filename)
    present = {path.name for path in (SOURCE / "model").iterdir()}
    if present != REQUIRED_MODEL_FILES:
        raise ValueError(
            f"모델 파일 구성이 다릅니다: missing={REQUIRED_MODEL_FILES - present}, "
            f"extra={present - REQUIRED_MODEL_FILES}"
        )


def build_zip() -> None:
    if OUTPUT.exists():
        OUTPUT.unlink()
    with zipfile.ZipFile(
        OUTPUT,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        archive.write(SOURCE / "script.py", "script.py")
        archive.write(SOURCE / "requirements.txt", "requirements.txt")
        for path in sorted((SOURCE / "model").iterdir()):
            archive.write(path, f"model/{path.name}")


def smoke_test() -> None:
    with tempfile.TemporaryDirectory(prefix="exp015-smoke-") as temporary:
        stage = Path(temporary)
        with zipfile.ZipFile(OUTPUT) as archive:
            archive.extractall(stage)
        (stage / "data").symlink_to(ROOT / "data", target_is_directory=True)
        result = subprocess.run(
            [str(PYTHON), "script.py"],
            cwd=stage,
            check=True,
            capture_output=True,
            text=True,
        )
        submission = stage / "output" / "submission.csv"
        if not submission.is_file():
            raise FileNotFoundError(submission)
        lines = submission.read_text(encoding="utf-8").splitlines()
        if len(lines) != 6:
            raise ValueError(f"샘플 제출 행 수가 잘못됐습니다: {len(lines)}")
        print(result.stdout.strip())


def inspect_zip() -> None:
    with zipfile.ZipFile(OUTPUT) as archive:
        names = archive.namelist()
        if any(name.startswith("data/") for name in names):
            raise ValueError("ZIP에 data 디렉터리가 포함되었습니다.")
        if names[:2] != ["script.py", "requirements.txt"]:
            raise ValueError(f"최상위 파일 구성이 잘못됐습니다: {names[:2]}")
        print("ZIP contents")
        for info in archive.infolist():
            print(
                f" {info.filename}: compressed={info.compress_size} "
                f"raw={info.file_size}"
            )


def main() -> None:
    validate_source()
    build_zip()
    inspect_zip()
    smoke_test()
    print(f"Ready: {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
