"""Findings and reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SEVERITIES = ("error", "warning", "info")


@dataclass
class Finding:
    check: str
    severity: str  # "error" | "warning" | "info"
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}, got {self.severity!r}")


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)

    def add(self, finding: Finding | None) -> None:
        if finding is not None:
            self.findings.append(finding)

    def extend(self, findings: list[Finding | None]) -> None:
        for f in findings:
            self.add(f)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "findings": [asdict(f) for f in self.findings]}

    def to_markdown(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        head = f"**Result: {status}** ({len(self.errors)} errors, {len(self.warnings)} warnings)\n"
        if not self.findings:
            return head + "\nNo findings.\n"
        icon = {"error": "ERROR", "warning": "WARN", "info": "INFO"}
        lines = ["| Severity | Check | Message |", "|---|---|---|"]
        for f in sorted(self.findings, key=lambda x: SEVERITIES.index(x.severity)):
            msg = f.message.replace("|", "\\|")
            lines.append(f"| {icon[f.severity]} | {f.check} | {msg} |")
        return head + "\n" + "\n".join(lines) + "\n"
