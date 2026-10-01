import json
from pathlib import Path

import pytest

from eegleak import run_metadata_checks
from eegleak.cli import main

EXAMPLES = Path(__file__).parents[1] / "examples"


def test_run_metadata_checks_clean(clean_df):
    report = run_metadata_checks(clean_df)
    assert report.ok and not report.warnings and len(report.findings) == 4


def test_run_metadata_checks_skips_optional_checks(clean_df):
    report = run_metadata_checks(clean_df.drop(columns=["start_s", "label"]))
    skipped = [f for f in report.findings if f.message.startswith("Skipped")]
    assert [f.check for f in skipped] == ["window_temporal_overlap", "label_shift"]


def test_run_metadata_checks_requires_id_columns(clean_df):
    with pytest.raises(ValueError, match="recording_id"):
        run_metadata_checks(clean_df.drop(columns="recording_id"))


def test_cli_exit_codes_and_markdown(capsys):
    assert main(["check", str(EXAMPLES / "clean_splits.csv")]) == 0
    assert capsys.readouterr().out.startswith("**Result: PASS**")
    assert main(["check", str(EXAMPLES / "leaky_splits.csv")]) == 1
    out = capsys.readouterr().out
    assert "**Result: FAIL** (3 errors, 1 warnings)" in out and "S05-N1" in out


def test_cli_json_and_options(tmp_path, capsys):
    assert main(["check", str(EXAMPLES / "leaky_splits.csv"), "--format", "json", "--gap-s", "30"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert not report["ok"] and {f["check"] for f in report["findings"]} >= {"subject_overlap", "label_shift"}
    csv = tmp_path / "renamed.csv"
    csv.write_text((EXAMPLES / "clean_splits.csv").read_text().replace("subject_id", "patient"))
    assert main(["check", str(csv), "--subject-col", "patient"]) == 0


def test_cli_missing_column_is_a_usage_error(tmp_path, capsys):
    csv = tmp_path / "bad.csv"
    csv.write_text("subject_id,split\nS1,train\n")
    with pytest.raises(SystemExit) as exc:
        main(["check", str(csv)])
    assert exc.value.code == 2 and "recording_id" in capsys.readouterr().err
