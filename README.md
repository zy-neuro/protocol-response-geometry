# protocol-response-geometry

Analysis code and derived tables for:

Yang Z. Flexible cellular phenotypes retain a shared type-level organization across response regimes. *eNeuro* (in press).

## Environment

Python 3.9 (NumPy 1.23, Matplotlib 3.9) on macOS (Apple Silicon). Figure regeneration:

```
pip install -r requirements.txt
```

SciPy is used only by `scripts/q3_mobility_modules.py`. Re-extracting split-half features from NWB needs IPFX / PyNWB. Reconstructing WGCNA modules needs R and WGCNA 1.74.

## Data

Raw data are not included.

- Electrophysiology: [DANDI 000020](https://dandiarchive.org/dandiset/000020/0.210913.1639), version `0.210913.1639`
- Transcriptome: [NeMO Allen Patch-seq](https://data.nemoarchive.org/other/AIBS/AIBS_patchseq/transcriptome/scell/SMARTseq/processed/analysis/20200611) release `20200611`

To re-run scripts that read those archives, place NWB files under `data/official/dandi_000020/` and the count matrix under `data/official/nemo/`. Derived tables used by the figures are already in `results/`.

## Usage

From this directory:

```
python scripts/figures_eneuro_results.py
python scripts/figures_eneuro_supplement.py
```

Analysis scripts are in `scripts/`. Random seed is 0.

## Citation

Please cite the paper above. This repository is the analysis code deposited as Extended Data 1. A journal citation with volume and DOI will be added when the article is published.

## License

MIT. See [LICENSE](LICENSE).
