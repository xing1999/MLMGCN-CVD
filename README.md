# MLMGCN-CVD
LLM-driven multimodal semantic graph learning framework for gut microbiome metagenomic DNA sequence-based ACVD risk prediction

## Highlights
- Multimodal node embeddings: DNABERT-S (semantic) + Nucleotide Transformer (context) + structure channel
- Two sub-models:
  - dsDNA-Model: structure channel uses DNAShape (dsDNA geometry)
  - ssDNA-Model: structure channel uses RNAfold/RNAplfold (ssDNA secondary-structure statistics)
- Decision-level screen–review strategy: dsDNA screening, ssDNA review on uncertain samples

## Repository structure (example)
- `Modeling/`: dataset & model definitions
- `scripts/`: training/inference scripts
- `data/`: feature files (note: large files may be managed via Git LFS or external links)
- `checkpoints/`: model weights (recommended: do not commit)

## Quick start
1. Prepare features under `data/test/feats/` and labels under `data/test/labels.csv`.
2. Run testing script for dsDNA/ssDNA or Screen–Review evaluation.

## Citation
If you use this code, please cite our paper.
