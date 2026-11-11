"""Test 8: the app runs, its banner comes from pitch.toml, and the shortfall flag responds to the inputs."""

import tomllib
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.report import SAMPLING_WARNING

ROOT = Path(__file__).resolve().parents[1]
PITCH = tomllib.loads((ROOT / "pitch.toml").read_text(encoding="utf-8"))


def run(**inputs):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    for key, value in inputs.items():
        widget = at.number_input(key) if key == "staffed" else at.slider(key)
        widget.set_value(value)
    if inputs:
        at.run()
    assert not at.exception, at.exception
    return at


def test_runs_and_banner_comes_from_pitch_toml():
    at = run()
    banner = PITCH["banner"]["text"].format(firm=PITCH["firm"]["name"])
    assert banner in [w.value for w in at.warning]
    assert [x.value for x in at.title] == ["Vulnerable customer capacity planner"]
    assert any("How many specialist handoffs does a catch-rate target need?" in m.value for m in at.markdown)
    assert any("· assumed" in s.label for s in at.slider)


def test_shortfall_renders_when_need_exceeds_staffed():
    at = run(staffed=50.0, target=95)
    assert any("Shortfall" in e.value for e in at.error)


def test_no_shortfall_when_staffed_is_ample():
    at = run(staffed=900.0, target=95)
    assert not any("Shortfall" in e.value for e in at.error)
    assert any("Within capacity" in s.value for s in at.success)


# The demo path and the hard ceiling (added 11 Nov 2026) -----------------------------------------------------------
def test_default_view_is_green_at_90():
    at = run()
    assert not any("Shortfall" in e.value for e in at.error)
    assert any("Within capacity" in s.value for s in at.success)


def test_default_panel_2_has_no_hard_ceiling():
    at = run()
    assert any("No hard ceiling at 150 per 1,000" in c.value for c in at.caption)
    assert not any("Hard ceiling" in e.value for e in at.error)


def test_panel_2_slider_is_a_what_if():
    at = run()
    assert "What if" in at.slider("capacity").label


def test_hard_ceiling_error_at_capacity_50_and_target_95():
    at = run(target=95, capacity=50)
    errors = [e.value for e in at.error if "Hard ceiling" in e.value]
    assert errors and "57.0" in errors[0] and "83.3%" in errors[0]


def test_ceiling_boundary_at_target_95():
    # 1,200 positives in 20,000: 95% needs 1,140 handoffs = 57.0 per 1,000 even with a perfect guardrail.
    at57 = run(target=95, capacity=57)
    assert not any("Hard ceiling" in e.value for e in at57.error)
    assert any("Hard ceiling at 57 per 1,000" in c.value for c in at57.caption)
    at56 = run(target=95, capacity=56)
    assert any("Hard ceiling at 56 per 1,000" in e.value for e in at56.error)


# The sampling warning (added 11 Nov 2026) ------------------------------------------------------------------------
def test_sampling_warning_renders_after_an_upload():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    at.radio("mode").set_value("Upload a CSV").run()
    sample = (ROOT / "data" / "sample_synthetic.csv").read_bytes()
    at.file_uploader("upload").set_value(("sample_synthetic.csv", sample, "text/csv")).run()
    assert not at.exception, at.exception
    assert [c.value for c in at.caption].count(SAMPLING_WARNING) == 1
    assert any("from your data" in m.label for m in at.metric)


def test_sampling_warning_absent_on_demo_data():
    at = run()
    assert SAMPLING_WARNING not in [c.value for c in at.caption]
