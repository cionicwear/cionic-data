#!/usr/bin/env python3
"""
Compare Jupyter notebook CONTENT between two directories (ignoring metadata/outputs
and benign IDs) and also compare CSV files.

Notebook compare:
- Matches notebooks by relative path
- Compares only cell sources (code/markdown/raw)
- Ignores outputs/metadata/execution counts
- Masks lines assigning datapath/download/files_url and any cionic/collections/<ID>

CSV compare:
- Normalizes newlines
- Trims whitespace in each cell
- By default compares in-order (row order matters)
- Optional --csv-unordered will sort rows (header kept first) to ignore row ordering

Usage:
  First, use a working branch containing these runnable scripts on a study of your choice (i.e. study 'khe', collection '13')
  From this branch, use runner.ipynb to run all of the scripts on this study, with 'Execute' checked and 'Overwrite' unchecked.
  Save a copy of the results to a new directory (i.e. copy recordings/cionic/khe/13 to recordings/cionic/khe/13 copy)
  Next, run the new branch containing these scripts on the same study.

  Finally, run this script from cionic-data using a command like:
  python tests/test_runnable_nbs.py "recordings/cionic/khe/13" "recordings/cionic/khe/13 copy" --diff

  It will show any differences between the old and new output notebooks / csvs.

Requires:
  pip install nbformat
"""

import argparse
import difflib
import csv
from pathlib import Path
import re
import nbformat

# ---------------- Notebook handling ----------------

DEFAULT_IGNORED_VARS = ["datapath", "download", "files_url"]

COLLECTION_ID_RE = re.compile(r"(cionic/collections/)[A-Za-z0-9_\-]+")

def mask_collection_ids(text: str) -> str:
    return COLLECTION_ID_RE.sub(r"\1<ID>", text)

def make_var_assign_re(ignored_vars):
    pattern = r"^\s*(" + "|".join(map(re.escape, ignored_vars)) + r")\s*=\s*.*$"
    return re.compile(pattern)

def mask_ignored_assignments(text: str, ignored_vars):
    assign_re = make_var_assign_re(ignored_vars) if ignored_vars else None
    out_lines = []
    for line in text.splitlines():
        if assign_re:
            m = assign_re.match(line)
            if m:
                varname = m.group(1)
                out_lines.append(f"{varname} = <IGNORED>")
                continue
        out_lines.append(line)
    return "\n".join(out_lines)

def list_files(root: Path, patterns):
    out = set()
    for pat in patterns:
        out |= {
            p.relative_to(root)
            for p in root.rglob(pat)
            if ".ipynb_checkpoints" not in p.parts
        }
    return out

def load_notebook(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return nbformat.read(f, as_version=4)

def normalize_notebook_content(nb, ignored_vars):
    blocks = []
    for i, cell in enumerate(nb.get("cells", [])):
        ctype = cell.get("cell_type", "")
        source = cell.get("source", "")
        if not isinstance(source, str):
            source = "\n".join(source)
        source = source.replace("\r\n", "\n").replace("\r", "\n")
        source = mask_collection_ids(source)
        source = mask_ignored_assignments(source, ignored_vars)
        blocks.append(f"### cell {i} [{ctype}] ###\n{source}\n")
    return blocks

def compare_notebooks(a_path: Path, b_path: Path, show_diff: bool, ignored_vars):
    nb_a = load_notebook(a_path)
    nb_b = load_notebook(b_path)
    a_blocks = normalize_notebook_content(nb_a, ignored_vars)
    b_blocks = normalize_notebook_content(nb_b, ignored_vars)
    if a_blocks == b_blocks:
        return True, None
    if show_diff:
        a_text = "".join(a_blocks).splitlines(keepends=False)
        b_text = "".join(b_blocks).splitlines(keepends=False)
        diff = difflib.unified_diff(
            a_text, b_text, fromfile=str(a_path), tofile=str(b_path), lineterm="", n=3
        )
        return False, "\n".join(diff)
    return False, None

# ---------------- CSV handling ----------------

def read_csv_normalized(path: Path):
    """
    Read CSV as rows of trimmed strings; normalize newlines & strip BOM.
    Returns (header_list, rows_list_of_tuples).
    """
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        rows = [[(c.strip() if isinstance(c, str) else c) for c in row] for row in reader]
    if not rows:
        return [], []
    header = rows[0]
    data = [tuple(r) for r in rows[1:]]
    return header, data

def compare_csv(a_path: Path, b_path: Path, show_diff: bool, unordered: bool):
    a_header, a_rows = read_csv_normalized(a_path)
    b_header, b_rows = read_csv_normalized(b_path)

    if a_header != b_header:
        if show_diff:
            diff = difflib.unified_diff(
                [",".join(a_header)], [",".join(b_header)],
                fromfile=str(a_path) + " (header)", tofile=str(b_path) + " (header)", lineterm=""
            )
            return False, "\n".join(diff)
        return False, None

    if unordered:
        a_rows_cmp = sorted(a_rows)
        b_rows_cmp = sorted(b_rows)
    else:
        a_rows_cmp = a_rows
        b_rows_cmp = b_rows

    if a_rows_cmp == b_rows_cmp:
        return True, None

    if show_diff:
        # Create diff-friendly string representations
        def to_lines(header, rows):
            yield ",".join(header)
            for r in rows:
                yield ",".join(r)
        a_lines = list(to_lines(a_header, a_rows_cmp))
        b_lines = list(to_lines(b_header, b_rows_cmp))
        diff = difflib.unified_diff(
            a_lines, b_lines, fromfile=str(a_path), tofile=str(b_path), lineterm="", n=3
        )
        return False, "\n".join(diff)
    return False, None

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(
        description="Compare notebooks (content-only) and CSVs between two directories."
    )
    ap.add_argument("dir_a", type=Path, help="Left directory")
    ap.add_argument("dir_b", type=Path, help="Right directory")
    ap.add_argument("--diff", action="store_true", help="Show unified diffs")
    ap.add_argument("--csv-unordered", action="store_true", help="Ignore CSV row order (sort rows)")
    ap.add_argument(
        "--ignore-var", action="append", default=[],
        help="Variable name to ignore in notebook assignments (repeatable). Defaults: datapath, download, files_url",
    )
    args = ap.parse_args()

    dir_a = args.dir_a.resolve()
    dir_b = args.dir_b.resolve()
    if not dir_a.is_dir() or not dir_b.is_dir():
        raise SystemExit("Both arguments must be directories.")

    ignored_vars = DEFAULT_IGNORED_VARS + args.ignore_var

    # Collect sets
    ipynb_a = list_files(dir_a, ["*.ipynb"])
    ipynb_b = list_files(dir_b, ["*.ipynb"])
    csv_a = list_files(dir_a, ["*.csv"])
    csv_b = list_files(dir_b, ["*.csv"])

    # Notebooks
    nb_only_a = sorted(ipynb_a - ipynb_b)
    nb_only_b = sorted(ipynb_b - ipynb_a)
    nb_both   = sorted(ipynb_a & ipynb_b)

    # CSVs
    csv_only_a = sorted(csv_a - csv_b)
    csv_only_b = sorted(csv_b - csv_a)
    csv_both   = sorted(csv_a & csv_b)

    nb_differ = []
    csv_differ = []

    # Compare notebooks
    for rel in nb_both:
        a_path, b_path = dir_a / rel, dir_b / rel
        same, diff = compare_notebooks(a_path, b_path, args.diff, ignored_vars)
        if not same:
            nb_differ.append(rel)
            print(f"NB DIFF: {rel}")
            if diff and args.diff:
                print(diff, "\n")

    # Compare CSVs
    for rel in csv_both:
        a_path, b_path = dir_a / rel, dir_b / rel
        same, diff = compare_csv(a_path, b_path, args.diff, args["csv_unordered"] if isinstance(args, dict) else args.csv_unordered)
        if not same:
            csv_differ.append(rel)
            print(f"CSV DIFF: {rel}")
            if diff and args.diff:
                print(diff, "\n")

    # Reports
    if nb_only_a:
        print(f"\nNotebooks only in LEFT ({dir_a}):")
        for r in nb_only_a: print(f"  {r}")
    if nb_only_b:
        print(f"\nNotebooks only in RIGHT ({dir_b}):")
        for r in nb_only_b: print(f"  {r}")

    if csv_only_a:
        print(f"\nCSVs only in LEFT ({dir_a}):")
        for r in csv_only_a: print(f"  {r}")
    if csv_only_b:
        print(f"\nCSVs only in RIGHT ({dir_b}):")
        for r in csv_only_b: print(f"  {r}")

    print("\nSummary:")
    print(f"  Notebooks compared: {len(nb_both)} | differing: {len(nb_differ)}")
    print(f"  CSVs compared:      {len(csv_both)} | differing: {len(csv_differ)}")
    if nb_differ:
        print("  Notebook files differing:")
        for r in nb_differ: print(f"    {r}")
    if csv_differ:
        print("  CSV files differing:")
        for r in csv_differ: print(f"    {r}")
    if (nb_differ or csv_differ) and not args.diff:
        print("\nUse --diff to see unified diffs.")

if __name__ == "__main__":
    main()
