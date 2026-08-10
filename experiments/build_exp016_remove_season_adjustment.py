"""EXP-015에서 2025 고정 보정만 제거한 EXP-016 ZIP을 만든다."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submissions" / "EXP-015"
OUTPUT = ROOT / "submit_exp016_no_season_adjustment.zip"
PYTHON = ROOT / ".venv" / "bin" / "python"


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="exp016-build-") as temporary:
        stage = Path(temporary)
        shutil.copy2(SOURCE / "requirements.txt", stage / "requirements.txt")
        shutil.copytree(SOURCE / "model", stage / "model")

        script = (SOURCE / "script.py").read_text(encoding="utf-8")
        old = "SEASON_2025_ADJUSTMENT = -0.005"
        new = "SEASON_2025_ADJUSTMENT = 0.0"
        if script.count(old) != 1:
            raise ValueError("EXP-015 보정 상수를 정확히 찾지 못했습니다.")
        (stage / "script.py").write_text(
            script.replace(old, new),
            encoding="utf-8",
        )

        metadata_path = stage / "model" / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["experiment"] = "EXP-016"
        metadata["ensemble"]["season_2025_adjustment"] = 0.0
        metadata["change_from_exp015"] = (
            "Remove only the fixed -0.005 season adjustment"
        )
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        (stage / "data").symlink_to(ROOT / "data", target_is_directory=True)
        result = subprocess.run(
            [str(PYTHON), "script.py"],
            cwd=stage,
            check=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout.strip())

        if OUTPUT.exists():
            OUTPUT.unlink()
        with zipfile.ZipFile(
            OUTPUT,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            archive.write(stage / "script.py", "script.py")
            archive.write(stage / "requirements.txt", "requirements.txt")
            for path in sorted((stage / "model").iterdir()):
                archive.write(path, f"model/{path.name}")

    with zipfile.ZipFile(OUTPUT) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"ZIP CRC 실패: {bad}")
        if any(name.startswith("data/") for name in archive.namelist()):
            raise ValueError("ZIP에 data가 포함됐습니다.")
    print(f"Ready: {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
