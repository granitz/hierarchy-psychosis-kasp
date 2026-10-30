"""
Build the de-identified clinical table from the study registry exports.

Replaces the seven ``pandas.read_clipboard()`` calls that the original
analysis relied on: instead of pasting each block out of Excel by hand, this
reads the same columns directly, joins them, keys the result on the scan
identifier and drops the clinical identifier entirely.

What comes out
--------------
``<CLINICAL_DIR>/clinical_data.csv`` with one row per scanned participant and
only these columns:

    ID1, Group, Is_Medicated, Gender, Age,
    Positive, Negative, General, Total, CGI_Score, Symptoms, Functioning

The clinical identifier (``ID0``), inclusion dates, birth years, handedness,
education and every other registry field are dropped. Nothing that could
identify a participant is written out.

Usage
-----
    python scripts/00_build_clinical_table.py
    python scripts/00_build_clinical_table.py --dry-run     # report, write nothing

Source files are expected in ``CLINICAL_DIR`` (see kasp/config.py). Each
registry export has the raw extract on the left and a tidied block on the
right; it is the tidied block this reads, which is the same block that was
being copied by hand.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import scipy.io as sio

from kasp import config

# --------------------------------------------------------------------------
# Where each field comes from.
#
#   file          filename within CLINICAL_DIR
#   header        0-based row holding the column names
#   columns       {source column name: name we want}
# --------------------------------------------------------------------------

SOURCES = {
    # Patients and controls are on separate sheets of the scan-mapping
    # workbook, and use different identifier formats: patients a number,
    # controls a letter-prefixed code. The patient sheet carries a tidied
    # ID0/ID1 block; the control sheet does not, so its raw columns are used.
    "id_map_patients": {
        "file": "KASP-MR.xls",
        "sheet": "Patienter",
        "header": 0,
        "columns": {"ID0": "ID0", "ID1": "ID1"},
    },
    "id_map_controls": {
        "file": "KASP-MR.xls",
        "sheet": "Kontroller",
        "header": 0,
        "columns": {"ID_0": "ID0", "ID_1": "ID1"},
    },
    "age_patients": {
        "file": "Bok10 *.xlsx",
        "header": 3,
        "columns": {"ID": "ID0", "Kön": "Gender", "Ålder": "Age"},
    },
    "age_controls": {
        "file": "Bok2 *.xlsx",
        "header": 3,
        "columns": {"ID": "ID0", "gender": "Gender", "age": "Age"},
    },
    "panss": {
        "file": "Bok8 *.xlsx",
        "header": 3,
        "columns": {"ID": "ID0", "sum_pos": "Positive", "sum_neg": "Negative",
                    "sum_general": "General", "sum_tot": "Total"},
    },
    "gaf_cgi": {
        "file": "Bok9 *.xlsx",
        "header": 3,
        "columns": {"ID": "ID0", "Sjukdomsgrad": "CGI_Score",
                    "symtom": "Symptoms", "function": "Functioning"},
    },
}

#: Columns allowed in the output. Anything else is dropped, by construction.
OUTPUT_COLUMNS = [
    "ID1", "Group", "Is_Medicated", "Gender", "Age",
    "Positive", "Negative", "General", "Total",
    "CGI_Score", "Symptoms", "Functioning",
]


def _read_block(spec, directory):
    """Read one tidied block, keeping only the columns we need.

    The registry exports repeat some column names on the left and right of the
    sheet (the raw extract and the tidied block). pandas disambiguates the
    duplicates with a ``.1`` suffix; the tidied block is the later occurrence,
    so the suffixed name is preferred when both are present.
    """
    # Registry exports are matched by their numbered prefix ("Bok8 *.xlsx");
    # the full export names carry personal names and dates that do not belong
    # in the code. Exactly one file must match.
    matches = sorted(Path(directory).glob(spec["file"]))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one file matching {spec['file']!r} in {directory}, "
            f"found {len(matches)}.")
    path = matches[0]

    frame = pd.read_excel(path, header=spec["header"],
                          sheet_name=spec.get("sheet", 0))
    out = pd.DataFrame()
    for source_name, target_name in spec["columns"].items():
        candidates = [c for c in frame.columns
                      if str(c) == source_name or str(c).startswith(f"{source_name}.")]
        if not candidates:
            raise KeyError(
                f"Column {source_name!r} not found in {spec['file']}. "
                f"Available: {list(frame.columns)}"
            )
        out[target_name] = frame[candidates[-1]]
    return out.dropna(how="all")


def _norm_id(series):
    """Normalise a clinical identifier to a comparable string.

    Patient identifiers are numeric and arrive from Excel as floats, so
    ``1234.0`` has to become ``"1234"``; control identifiers are
    letter-prefixed codes. Both are trimmed and upper-cased so the two sides
    of a merge agree.
    """
    def norm(value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value).strip().upper()
    return series.map(norm)


def _to_numeric_gender(series):
    """Map the sex column to 0 = man, 1 = woman, as the original script did."""
    mapped = series.replace({"Man": 0, "Kvinna": 1, "man": 0, "kvinna": 1})
    return pd.to_numeric(mapped, errors="coerce")


def _cgi_to_number(series):
    """'5 Påtagligt sjuk' -> 5. Blank or unparseable entries become NaN."""
    def parse(value):
        if isinstance(value, str) and value.split():
            head = value.split()[0]
            return int(head) if head.isdigit() else np.nan
        return pd.to_numeric(value, errors="coerce")
    return series.map(parse)


def build(clinical_dir=None, base_dir=None):
    """Assemble the de-identified table. Returns a DataFrame."""
    clinical_dir = Path(clinical_dir or config.CLINICAL_DIR)

    def scan_map(key):
        frame = _read_block(SOURCES[key], clinical_dir).dropna(subset=["ID0", "ID1"])
        frame["ID0"] = _norm_id(frame["ID0"])
        frame["ID1"] = ["sub-" + str(int(float(v))) for v in frame["ID1"]]
        return frame.dropna(subset=["ID0"])

    id_map_sz = scan_map("id_map_patients")
    id_map_ctrl = scan_map("id_map_controls")

    patients = _read_block(SOURCES["age_patients"], clinical_dir)
    controls = _read_block(SOURCES["age_controls"], clinical_dir)
    panss = _read_block(SOURCES["panss"], clinical_dir)
    gaf = _read_block(SOURCES["gaf_cgi"], clinical_dir)

    for frame in (patients, controls, panss, gaf):
        frame["ID0"] = _norm_id(frame["ID0"])

    for frame in (patients, controls):
        frame["Gender"] = _to_numeric_gender(frame["Gender"])
        frame["Age"] = pd.to_numeric(frame["Age"], errors="coerce")

    gaf["CGI_Score"] = _cgi_to_number(gaf["CGI_Score"])
    for col in ("Symptoms", "Functioning"):
        gaf[col] = pd.to_numeric(gaf[col], errors="coerce")
    for col in ("Positive", "Negative", "General", "Total"):
        panss[col] = pd.to_numeric(panss[col], errors="coerce")

    # Which scans entered the analyses, and which patients were medicated.
    from kasp import data as kdata

    controls_scans, patient_scans, is_med = kdata.load_subject_lists()

    # Patients: demographics + PANSS + GAF/CGI, then attach the scan id.
    df_sz = patients.merge(panss, on="ID0", how="left").merge(gaf, on="ID0", how="left")
    df_sz = df_sz.merge(id_map_sz, on="ID0", how="left")
    df_sz = df_sz[df_sz["ID1"].isin(patient_scans)]
    df_sz["Group"] = "FEP"

    med_lookup = pd.DataFrame({"ID1": patient_scans, "Is_Medicated": is_med})
    df_sz = df_sz.merge(med_lookup, on="ID1", how="left")

    # Controls: demographics only.
    df_ctrl = controls.merge(id_map_ctrl, on="ID0", how="left")
    df_ctrl = df_ctrl[df_ctrl["ID1"].isin(controls_scans)]
    df_ctrl["Group"] = "Control"
    df_ctrl["Is_Medicated"] = np.nan

    combined = pd.concat([df_sz, df_ctrl], ignore_index=True)

    # Drop the clinical identifier and anything else not explicitly allowed.
    for col in OUTPUT_COLUMNS:
        if col not in combined.columns:
            combined[col] = np.nan
    combined = combined[OUTPUT_COLUMNS]
    combined = combined.drop_duplicates(subset=["ID1"]).sort_values("ID1")

    return combined.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clinical-dir", default=None,
                        help="directory holding the registry exports "
                             "(default: CLINICAL_DIR from kasp/config.py)")
    parser.add_argument("--output", default=None,
                        help="output CSV (default: <CLINICAL_DIR>/clinical_data.csv)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be written, write nothing")
    args = parser.parse_args()

    clinical_dir = Path(args.clinical_dir or config.CLINICAL_DIR)
    output = Path(args.output or clinical_dir / "clinical_data.csv")

    table = build(clinical_dir)

    n_ctrl = (table["Group"] == "Control").sum()
    n_unmed = ((table["Group"] == "FEP") & (table["Is_Medicated"] == 0)).sum()
    n_med = ((table["Group"] == "FEP") & (table["Is_Medicated"] == 1)).sum()

    print(f"rows: {len(table)}  "
          f"(controls {n_ctrl}, FEP unmedicated {n_unmed}, FEP medicated {n_med})")
    print(f"columns: {list(table.columns)}")
    print("\nnon-null counts per column:")
    for col in table.columns:
        print(f"   {col:<14} {table[col].notna().sum():3d} / {len(table)}")

    assert "ID0" not in table.columns, "clinical identifier leaked into the output"

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(f"\nwritten -> {output}")
    print("This file contains participant data. Keep it out of version control; "
          "the repository's .gitignore already excludes it.")


if __name__ == "__main__":
    main()
