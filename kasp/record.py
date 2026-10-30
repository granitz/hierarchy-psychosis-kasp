"""
A flat record of every statistic a run reports.

The notebook calls :meth:`StatsLog.add` next to each number it prints, and
writes the log once at the end to ``runs/<KASP_RUN_TAG>/stats.csv``. That
file is small, group-level and plain text, so it is committed: comparing two
runs' ``stats.csv`` is how a change in the pipeline shows up as a change in the
numbers.

Only group-level values belong here. :meth:`StatsLog.add` refuses anything
that is not a single finite-or-NaN number, which keeps per-participant arrays
out by construction.

Long format, one row per number::

    section, label, quantity, value

``section`` is where in the paper it comes from (``figure2``, ``directedness``),
``label`` which group or contrast, ``quantity`` what it is (``r``, ``p_spin``,
``U``). The triple is unique within a run; recording it again (re-running a
cell) replaces the earlier value.
"""

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["StatsLog", "read_stats"]

COLUMNS = ["section", "label", "quantity", "value"]


def _scalar(value):
    """A plain float, or an error for anything that is not one number."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("booleans are derived from other values; record those instead")
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(
            f"only single group-level values are recorded, got shape {arr.shape}"
        )
    return float(arr.ravel()[0])


def _slug(text):
    """Stable, CSV-friendly identifier from a display label."""
    text = str(text).strip()
    for old, new in [(" vs ", "_vs_"), (" - ", "_minus_"), (" / ", "__"),
                     ("$", ""), ("\\", ""),
                     ("(", ""), (")", ""), (" ", "_"), ("|", ""), (",", "")]:
        text = text.replace(old, new)
    return text


class StatsLog:
    def __init__(self):
        self._rows = {}

    # ------------------------------------------------------------------ add
    def add(self, section, quantity, value, label=""):
        """Record one number."""
        key = (str(section), _slug(label), _slug(quantity))
        # Re-running a notebook cell records the same number again; the latest
        # value wins, which is what the final write should reflect.
        self._rows[key] = _scalar(value)

    def add_frame(self, section, frame, label_columns=None):
        """Record every numeric cell of a small results table.

        Index and string columns become the label; numeric columns become
        quantities; boolean columns are skipped, as they are thresholds applied
        to numbers already recorded.
        """
        flat = frame.reset_index()
        if label_columns is None:
            label_columns = [c for c in flat.columns
                             if flat[c].dtype == object and c != "index"]
        numeric = [c for c in flat.columns
                   if c not in label_columns and c != "index"
                   and pd.api.types.is_numeric_dtype(flat[c])
                   and not pd.api.types.is_bool_dtype(flat[c])]
        for _, row in flat.iterrows():
            label = " / ".join(str(row[c]) for c in label_columns)
            for column in numeric:
                self.add(section, column, row[column], label=label)

    # -------------------------------------------------------------- output
    def to_frame(self):
        rows = [(*key, value) for key, value in self._rows.items()]
        return pd.DataFrame(rows, columns=COLUMNS)

    def write(self, run_dir, run_tag, inputs=None):
        """Write ``stats.csv`` and ``manifest.json`` into ``run_dir``.

        The manifest ties the numbers to the code that produced them: the git
        commit, whether the working tree was clean, and the run tag. A run from
        a dirty tree is still written, but flagged -- its numbers cannot be
        traced to a commit.
        """
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        frame = self.to_frame()
        # repr-precision floats, so a rerun can be compared at machine precision
        frame.to_csv(run_dir / "stats.csv", index=False, float_format="%.17g")

        manifest = {
            "run_tag": run_tag,
            "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git": _git_state(run_dir),
            "python": platform.python_version(),
            "n_values": int(len(frame)),
            "n_nan": int(frame["value"].isna().sum()),
            "inputs": inputs or {},
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return frame


def _git_state(where):
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=where, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:
            return None

    # Tracked files only. Untracked files -- this run's own stats.csv among
    # them -- say nothing about which code produced the numbers.
    status = run("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": run("rev-parse", "HEAD"),
        "describe": run("describe", "--tags", "--always", "--dirty"),
        "clean": (status == "") if status is not None else None,
    }


def read_stats(path):
    """Read a ``stats.csv`` back without losing precision.

    pandas' default float parser is fast but not exact; ``round_trip`` returns
    the same doubles that were written, which a machine-precision comparison
    needs.
    """
    return pd.read_csv(path, float_precision="round_trip",
                       dtype={"section": str, "label": str, "quantity": str},
                       keep_default_na=False, na_values=["", "nan", "NaN"])
