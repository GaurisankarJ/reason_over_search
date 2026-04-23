from pathlib import Path


def test_no_verl_legacy_runtime_imports():
    root = Path(__file__).resolve().parents[1]
    checked = [
        root / "run_eval.py",
        root / "flashrag" / "pipeline" / "__init__.py",
        root / "flashrag" / "pipeline" / "active_pipeline.py",
    ]
    for file_path in checked:
        content = file_path.read_text(encoding="utf-8")
        assert "verl_legacy" not in content

