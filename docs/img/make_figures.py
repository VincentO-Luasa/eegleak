"""Generate the README figures (SVG, light and dark mode). Run from the repository root:

python docs/img/make_figures.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent
STYLE = """<style>
svg { --bg: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --rule: #d9d8d3; --hl: #c62f2f;
      --train: #2a78d6; --val: #1baf7a; --test: #eb6834; }
@media (prefers-color-scheme: dark) {
  svg { --bg: #1a1a19; --ink: #ffffff; --ink2: #c3c2b7; --rule: #3d3d3a; --hl: #e66767;
        --train: #3987e5; --val: #199e70; --test: #d95926; }
}
text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
       font-size: 13px; fill: var(--ink); }
.bg { fill: var(--bg); } .muted { fill: var(--ink2); font-size: 12px; } .h { font-weight: 600; font-size: 15px; }
.train { fill: var(--train); } .val { fill: var(--val); } .test { fill: var(--test); }
.rule { stroke: var(--rule); fill: none; } .hl { stroke: var(--hl); fill: none; stroke-width: 2; }
.hlt { fill: var(--hl); font-weight: 600; } .trace { stroke: var(--ink2); fill: none; stroke-width: 1.2; }
</style>"""


def svg(name, width, height, body):
    (OUT / name).write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f"{STYLE}<rect class='bg' width='{width}' height='{height}' rx='8'/>{''.join(body)}</svg>\n"
    )


def text(x, y, s, cls="", anchor="start"):
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{s}</text>'


def legend(x, y, splits=("train", "val", "test")):
    return [rect(x + 70 * i, y - 10, 12, 12, s) + text(x + 70 * i + 18, y, s) for i, s in enumerate(splits)]


def tick(x):
    return f'<line class="rule" x1="{x}" y1="50" x2="{x}" y2="150" stroke-dasharray="3 3"/>'


def rect(x, y, w, h, cls, title=""):
    tip = f"<title>{title}</title>" if title else ""
    return f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{h}" rx="3" class="{cls}">{tip}</rect>'


def leakage():
    """Window-level vs subject-level split of the same four people."""
    rng = np.random.default_rng(1)
    people, n = "ABCD", 12
    body = legend(560, 30, ("train", "test"))
    leaky = ["✗ Every test window comes from a person the model", "already trained on: it can recognise the person,"]
    leaky.append("so the test score is inflated.")
    correct = ["✓ The test person is new to the model, so the score", "shows how it would do on a new patient."]
    panels = [
        (20, "Split by window (leaky)", rng.random((4, n)) < 0.3, leaky),
        (400, "Split by person (correct)", np.array([[p == "D"] * n for p in people]), correct),
    ]
    for x0, title, is_test, verdict in panels:
        body.append(text(x0, 60, title, "h"))
        for r, person in enumerate(people):
            y = 80 + 28 * r
            body.append(text(x0, y + 14, f"Person {person}", "muted"))
            body += [rect(x0 + 70 + 22 * c, y, 18, 18, "test" if t else "train") for c, t in enumerate(is_test[r])]
        body += [text(x0, 210 + 18 * i, line) for i, line in enumerate(verdict)]
    body.append(text(20, 280, "Each square is one window: a few seconds of EEG from one person.", "muted"))
    svg("leakage.svg", 760, 296, body)


def windows():
    """A recording is cut into fixed-length windows; each window is one row of the splits table."""
    t = np.linspace(0, 180, 900)
    rng = np.random.default_rng(0)
    signal = np.sin(2 * np.pi * t / 7) + 0.6 * np.sin(2 * np.pi * t / 1.3) + 0.5 * rng.normal(size=t.size)
    x0, scale = 40, 4.0  # px per second
    pts = " ".join(f"{x0 + a * scale:.1f},{88 - 11 * b:.1f}" for a, b in zip(t, signal, strict=True))
    labels = ["W", "W", "N1", "N2", "N2", "N2"]
    body = [
        text(x0, 30, "One recording (subject S01, night S01-N1), cut into 30-second windows", "h"),
        f'<polyline class="trace" points="{pts}"/>',
    ]
    for k, label in enumerate(labels):
        x = x0 + 30 * k * scale
        body += [tick(x), text(x + 60, 140, f"window {k}", "muted", "middle"), text(x + 60, 158, label, "", "middle")]
    body += [
        tick(x0 + 720),
        text(x0, 190, "Illustrative signal and labels. eegleak only needs this table, one row per window:", "muted"),
    ]
    cols = ["subject_id", "recording_id", "split", "start_s", "end_s", "label"]
    rows = [["S01", "S01-N1", "train", f"{30.0 * k}", f"{30.0 * (k + 1)}", lab] for k, lab in enumerate(labels[:3])]
    for r, row in enumerate([cols, *rows, ["…"] * 6]):
        y = 220 + 22 * r
        body += [text(x0 + 125 * c, y, v, "h" if r == 0 else "") for c, v in enumerate(row)]
    body.append(f'<line class="rule" x1="{x0}" y1="227" x2="{x0 + 740}" y2="227"/>')
    svg("windows.svg", 820, 340, body)


def example_splits(csv="examples/leaky_splits.csv"):
    """Timeline of every recording in the leaky example, coloured by split, with the planted problems."""
    df = pd.read_csv(csv).sort_values(["recording_id", "start_s"])
    x0, width, top, row_h = 80, 540, 70, 16
    scale = width / df["end_s"].max()
    recs = list(dict.fromkeys(df["recording_id"]))
    body = [text(20, 28, "examples/leaky_splits.csv: one bar per recording, coloured by split", "h"), *legend(x0, 52)]
    for i, (rec, g) in enumerate(df.groupby("recording_id", sort=False)):
        y = top + row_h * i
        body.append(text(x0 - 8, y + 11, rec, "muted", "end"))
        runs = g.groupby((g["split"] != g["split"].shift()).cumsum())
        for _, run in runs:
            s, e, split = run["start_s"].min(), run["end_s"].max(), run["split"].iloc[0]
            title = f"{rec} {split}: {s:.0f}–{e:.0f} s, {len(run)} windows"
            body.append(rect(x0 + s * scale, y, (e - s) * scale - 1, row_h - 4, split, title))
    y_axis = top + row_h * len(recs) + 6
    for s in range(0, 1201, 300):
        body.append(text(x0 + s * scale, y_axis + 12, f"{s} s", "muted", "middle"))

    def note(rec, lines, x=None):
        y = top + row_h * recs.index(rec) + 11
        return [text(x0 + width + 16, y + 15 * k, line, "hlt" if k == 0 else "muted") for k, line in enumerate(lines)]

    y = top + row_h * recs.index("S05-N1") - 2
    x, w = x0 + 590 * scale, 35 * scale  # the 600-615 s overlap, padded to stay visible
    body += [
        f'<rect class="hl" x="{x:.1f}" y="{y}" width="{w:.1f}" height="{row_h}" rx="3"/>',
        *note("S03-N1", ["1. Subject S03: night 1 in train,", "night 2 in test"]),
        *note("S05-N1", ["2. Recording cut at 600 s; its", "windows overlap across the cut"]),
        *note("S11-N1", ["3. Test subjects mostly awake", "(label shift)"]),
    ]
    svg("example_splits.svg", 880, y_axis + 30, body)


if __name__ == "__main__":
    leakage()
    windows()
    example_splits()
