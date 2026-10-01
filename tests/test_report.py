import pytest

from eegleak import Finding, Report


def test_invalid_severity():
    with pytest.raises(ValueError):
        Finding("x", "fatal", "bad")


def test_report_ok_and_rendering():
    r = Report()
    assert r.ok and "No findings" in r.to_markdown()
    r.extend([Finding("a", "warning", "w"), None, Finding("b", "error", "e | pipe")])
    assert not r.ok and len(r.errors) == 1 and len(r.warnings) == 1
    md = r.to_markdown()
    assert md.startswith("**Result: FAIL** (1 errors, 1 warnings)")
    assert md.index("ERROR") < md.index("WARN") and "e \\| pipe" in md
    assert r.to_dict()["findings"][0] == {"check": "a", "severity": "warning", "message": "w", "details": {}}
