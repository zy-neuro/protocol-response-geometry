"""Revision-copy paths.

Read submitted phenotype / join / Q2–Q3 tables from the original `results/`.
Write new revision outputs under `revision/results` and `revision/figures`.
"""

from pathlib import Path

REV = Path(__file__).resolve().parents[1]
REPO = REV.parent

ORIG_RESULTS = REPO / "results"
ORIG_PHENO = ORIG_RESULTS / "phenotype"
ORIG_FEATURES = ORIG_RESULTS / "features"
ORIG_JOIN = ORIG_RESULTS / "m0" / "m0_molecular_join.csv"

OUT_RESULTS = REV / "results"
OUT_PHENO = OUT_RESULTS / "phenotype"
OUT_Q1 = OUT_RESULTS / "q1"
OUT_Q1_REL = OUT_RESULTS / "q1_reliability"
OUT_Q3_MOB = OUT_RESULTS / "q3_mobility"
OUT_FIG = REV / "figures"
OUT_FIG_MAIN = OUT_FIG / "results"
OUT_FIG_SUPP = OUT_FIG / "supplement"
OUT_TAB = REV / "tables"
