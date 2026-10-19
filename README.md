# Extended Data 1

Analysis code and derived tables for the accompanying manuscript. Work from
the directory that contains this README.

Analysis scripts reproduce the reported analyses from the publicly available
source data / specified inputs.

Figure scripts reproduce all manuscript and Extended Data figures from the
derived result tables included in this archive.

This archive is not a self-contained raw-data → figure pipeline. NWB files
and the NeMO count matrix are not included.

## Source data (public archives; not duplicated)

- Electrophysiology: DANDI dandiset 000020, version `0.210913.1639`
  https://dandiarchive.org/dandiset/000020/0.210913.1639
  DOI: `10.48324/dandi.000020/0.210913.1639`
- Transcriptome: NeMO Allen Institute Patch-seq release `20200611`
  count matrix `20200513_Mouse_PatchSeq_Release_count.v2.csv`
  metadata `20200625_patchseq_metadata_mouse.csv`
  https://data.nemoarchive.org/other/AIBS/AIBS_patchseq/transcriptome/scell/SMARTseq/processed/analysis/20200611

To re-run scripts that need those archives, place them at
`data/official/dandi_000020/` (`*_icephys.nwb`) and `data/official/nemo/`.
Cell identifiers in the tables are public DANDI / Allen Patch-seq specimen names.

## Environment

Reported analyses were run with Python 3.9 (NumPy 1.23, Matplotlib 3.9,
SciPy 1.13) and R 4.5 with WGCNA 1.74. A full conda/Docker lockfile is not
included.

Figure regeneration needs only NumPy and Matplotlib:

```
pip install -r requirements.txt
```

SciPy is used by the Python module-association helper
(`revision/scripts/q3_mobility_modules.py`). Re-extracting split-half
features from NWB additionally needs IPFX / PyNWB. Reconstructing WGCNA
modules additionally needs R and the WGCNA package (v1.74).

## Random seeds

Stochastic procedures use seed = 0 (`numpy.random.default_rng(0)` or
`set.seed(0)` / WGCNA `randomSeed`). Split-half fallback (when odd/even
sweep numbers cannot be used) is also seed 0.

| Procedure | Replicates | Script |
|-----------|------------|--------|
| Pairwise *D* / 15-NN chance | 100 index shuffles | `scripts/q1_describe_phenotype.py` |
| Neighborhood-size chance | 100 | `revision/scripts/q1_knn_scale.py` |
| Split-half chance | 100 | `revision/scripts/q1_split_half_long_square.py` |
| Q2 alignment null | 100 | `scripts/q2_anchor.py` |
| Q3a *S* tags / concordance | 100 | `scripts/q3a_explain.py` |
| Equal-N draws; type-label *S* null | 200 draws; 1,000 shuffles | `scripts/q3a_centroid_robustness.py` |
| Q3b/c panel and module tests | 50 | `scripts/q3bc_explain.py` |
| Mobility vs module eigengene | 1,000 | `revision/scripts/q3_wgcna_mobility.py` |

## Which scripts need what

**Figure scripts (bundled tables only; no NWB / no NeMO)**

```
python revision/scripts/figures_eneuro_results.py
python revision/scripts/figures_eneuro_supplement.py
```

| Script output | Manuscript |
|---------------|------------|
| `revision/figures/results/Fig1_describe` | Figure 1 |
| `revision/figures/results/Fig2_anchor_2x2` | Figure 2 |
| `revision/figures/results/Fig3_compare_backbone` | Figure 3 |
| `revision/figures/results/Fig4_explain_shared` | Figure 4 |
| `revision/figures/supplement/Figure 1-1` | Extended Data Figure 1-1 |
| `revision/figures/supplement/Figure 1-2` | Extended Data Figure 1-2 |
| `revision/figures/supplement/Figure 1-3` | Extended Data Figure 1-3 |
| `revision/figures/supplement/Figure 2-1` | Extended Data Figure 2-1 |
| `revision/figures/supplement/Figure 3-2` | Extended Data Figure 3-2 (S maps + tags) |
| `revision/figures/supplement/Figure 3-3` | Extended Data Figure 3-3 (equal-N + type-level split-half S) |
| `revision/figures/supplement/Figure 4-2` | Extended Data Figure 4-2 (RNA panels + markers) |
| `revision/figures/supplement/Figure 4-4` | Extended Data Figure 4-4 (mobility + WGCNA) |

Word-format Extended Data (not written by these scripts): Figure 3-1 (t-type counts), Figure 4-1 (module gene sets), Figure 4-3 (WGCNA–mobility ρ).

Figure 1A uses `results/fig1_example_traces.csv` (DANDI session 639797293).

**Analysis scripts that recompute from bundled derived tables**

Phenotype PCA scores, variance tables, and the molecular join are included.
These do not need NWB or NeMO:

- `scripts/q1_describe_phenotype.py`
- `revision/scripts/q1_knn_scale.py`
- `scripts/q2_anchor.py`
- `scripts/q3a_explain.py`
- `scripts/q3a_centroid_robustness.py`

PC1 loading tables are included under `revision/results/phenotype/`.
Re-dumping them (`revision/scripts/dump_pc1_loadings.py`) needs the original
IPFX feature table (`results/features/`), which is not bundled.

**Analysis scripts that need the public archives**

- Long-square split-half extract: `revision/scripts/q1_split_half_long_square.py`
  (DANDI 000020 NWB + IPFX). The tables used in Extended Data Figure 1-3
  are already included.
- Q3b/c RNA geometry and WGCNA: `scripts/q3bc_explain.py`,
  `revision/scripts/q3_wgcna_mobility.py`, `q3_wgcna_modules.R`
  (NeMO count matrix). The tables used in Figure 4 and Extended Data
  Figure 4-4 are already included.

Prespecified excitability genes: `configs/q3_curated_excitability_genes.txt`.

## Analyses and tables

| Analysis | Code | Tables used in the figures |
|----------|------|----------------------------|
| Phenotype spaces, pairwise *D*, 15-NN | `scripts/q1_describe_phenotype.py` | `results/phenotype/`, `results/q1/` |
| Neighborhood-size sweep | `revision/scripts/q1_knn_scale.py` | `revision/results/q1/` |
| Long-square split-half neighborhoods | `revision/scripts/q1_split_half_long_square.py` | `revision/results/q1_reliability/` |
| PC1 loadings | `revision/scripts/dump_pc1_loadings.py` | `revision/results/phenotype/` |
| t-type / Cre anchoring | `scripts/q2_anchor.py` | `results/q2/` |
| Type–type *S* maps, tags, equal-N | `scripts/q3a_explain.py`, `q3a_centroid_robustness.py` | `results/q3a/`, `results/q3a_robustness/` |
| Transcriptomic *T* vs mean *S* / rank-range | `scripts/q3bc_explain.py` | `results/q3b/`, `results/q3c/` |
| Cell-level mobility + WGCNA | `revision/scripts/q3_wgcna_mobility.py`, `q3_mobility_modules.py`, `q3_wgcna_modules.R` | `revision/results/q3_mobility/` |
