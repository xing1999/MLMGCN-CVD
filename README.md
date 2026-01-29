# MLMGCN-CVD
LLM-driven multimodal semantic graph learning framework for gut microbiome metagenomic DNA sequence-based ACVD risk prediction

## Highlights
- Multimodal node embeddings: DNABERT-S (semantic) + Nucleotide Transformer (context) + structure channel
- Two sub-models:
  - dsDNA-Model: structure channel uses DNAShape (dsDNA geometry)
  - ssDNA-Model: structure channel uses RNAfold/RNAplfold (ssDNA secondary-structure statistics)
- Decision-level screen–review strategy: dsDNA screening, ssDNA review on uncertain samples

## Quick start
1. Prepare features under `data/test/feats/` and labels under `data/test/labels.csv`.
2. Run testing script for dsDNA/ssDNA or Screen–Review evaluation.
