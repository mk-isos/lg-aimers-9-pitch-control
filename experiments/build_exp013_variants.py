"""EXP-013 모델을 재사용해 리더보드 분해 실험 ZIP을 만든다."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submissions" / "EXP-013"
DATA = ROOT / "data"

VARIANTS = {
    "raw": {
        "catboost_weight": 0.28719567,
        "lightgbm_weight": 0.71280433,
        "scale": 1.0,
        "intercept": 0.0,
    },
    "lgb_calibrated": {
        "catboost_weight": 0.0,
        "lightgbm_weight": 1.0,
        "scale": 1.103691565923087,
        "intercept": -0.062343068328920795,
    },
    "offset_down": {
        "catboost_weight": 0.28719567,
        "lightgbm_weight": 0.71280433,
        "scale": 1.12708208,
        "intercept": -0.07836118,
    },
    "offset_up": {
        "catboost_weight": 0.28719567,
        "lightgbm_weight": 0.71280433,
        "scale": 1.12708208,
        "intercept": -0.06836118,
    },
}


def replace_constant(source: str, name: str, value: float) -> str:
    pattern = rf"^{name}\s*=\s*[-+0-9.eE]+$"
    replacement = f"{name} = {value!r}"
    updated, replacements = re.subn(
        pattern,
        replacement,
        source,
        count=1,
        flags=re.MULTILINE,
    )
    if replacements != 1:
        raise ValueError(f"{name} 상수를 정확히 한 번 교체하지 못했습니다.")
    return updated


def build_variant(name: str, config: dict[str, float]) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix=f"exp013-{name}-") as temp_name:
        stage = Path(temp_name)
        shutil.copytree(SOURCE / "model", stage / "model")
        shutil.copy2(SOURCE / "requirements.txt", stage / "requirements.txt")
        script = (SOURCE / "script.py").read_text(encoding="utf-8")
        replacements = {
            "CATBOOST_WEIGHT": config["catboost_weight"],
            "LIGHTGBM_WEIGHT": config["lightgbm_weight"],
            "CALIBRATION_SCALE": config["scale"],
            "CALIBRATION_INTERCEPT": config["intercept"],
        }
        for constant, value in replacements.items():
            script = replace_constant(script, constant, value)
        (stage / "script.py").write_text(script, encoding="utf-8")

        data_dir = stage / "data"
        data_dir.mkdir()
        shutil.copy2(DATA / "test.csv", data_dir / "test.csv")
        shutil.copy2(
            DATA / "sample_submission.csv",
            data_dir / "sample_submission.csv",
        )
        subprocess.run(
            [sys.executable, "script.py"],
            cwd=stage,
            check=True,
            capture_output=True,
            text=True,
        )
        submission = pd.read_csv(stage / "output" / "submission.csv")
        if list(submission.columns) != ["row_id", "control_success"]:
            raise ValueError(f"{name}: 제출 컬럼이 올바르지 않습니다.")
        if submission["control_success"].isna().any():
            raise ValueError(f"{name}: 예측에 결측값이 있습니다.")
        if not submission["control_success"].between(0, 1).all():
            raise ValueError(f"{name}: 예측값이 0~1 범위를 벗어났습니다.")

        zip_path = ROOT / f"submit_exp013_{name}.zip"
        with zipfile.ZipFile(
            zip_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            archive.write(stage / "script.py", "script.py")
            archive.write(stage / "requirements.txt", "requirements.txt")
            for model_path in sorted((stage / "model").iterdir()):
                if model_path.is_file():
                    archive.write(model_path, f"model/{model_path.name}")
        return {
            "variant": name,
            "zip": str(zip_path),
            "zip_bytes": zip_path.stat().st_size,
            "prediction_mean": float(submission["control_success"].mean()),
            "prediction_min": float(submission["control_success"].min()),
            "prediction_max": float(submission["control_success"].max()),
            **config,
        }


def main() -> None:
    results = [build_variant(name, config) for name, config in VARIANTS.items()]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
