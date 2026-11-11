"""Test 9: no firm is named in app.py or src/. Firm-specific text lives only in pitch.toml."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_firm_name_only_in_pitch_toml():
    firm = tomllib.loads((ROOT / "pitch.toml").read_text(encoding="utf-8"))["firm"]["name"].lower()
    files = [ROOT / "app.py", *sorted((ROOT / "src").glob("*.py"))]
    offenders = [f.name for f in files if firm in f.read_text(encoding="utf-8").lower()]
    assert offenders == []
