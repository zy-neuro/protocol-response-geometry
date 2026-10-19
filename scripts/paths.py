"""Shared paths. Repo root is the parent of this scripts/ directory."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REV = REPO
RESULTS = REPO / "results"

ORIG_RESULTS = RESULTS
ORIG_PHENO = RESULTS / "phenotype"
ORIG_FEATURES = RESULTS / "features"
ORIG_JOIN = RESULTS / "m0" / "m0_molecular_join.csv"

OUT_RESULTS = RESULTS
OUT_PHENO = RESULTS / "phenotype"
OUT_Q1 = RESULTS / "q1"
OUT_Q1_REL = RESULTS / "q1_reliability"
OUT_Q3_MOB = RESULTS / "q3_mobility"
OUT_FIG = REPO / "figures"
OUT_FIG_MAIN = OUT_FIG / "results"
OUT_FIG_SUPP = OUT_FIG / "supplement"
OUT_TAB = REPO / "tables"
