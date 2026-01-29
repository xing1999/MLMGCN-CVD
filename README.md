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


## Availability
- Source code and scripts: this repository.
- Web server (optional): please fill in your deployed URL if available.


Requirements
Tested environment (recommended):

Python >= 3.9

PyTorch (GPU optional)

numpy, pandas, scikit-learn

A minimal setup:

bash
复制代码
pip install -r requirements.txt
If you do not have requirements.txt yet, create one based on your environment:
pip freeze > requirements.txt

Repository Structure (suggested)
powershell

MLMGCN-CVD/
├── data/
│   └── AHT/
│       ├── labels.csv
│       └── feats/
│           ├── dnaberts_attn/
│           ├── dnaberts_emb/
│           ├── dnashape/       # dsDNA structure features
│           ├── ss_struct/      # ssDNA structure features
│           ├── nt_emb_seq/
│           └── nt_emb_tok/
├── checkpoints/
│   ├── dsDNA-Model/
│   └── ssDNA-Model/
├── Modeling/                   # Dataset + Model definitions
├── scripts/                    # training / evaluation scripts
└── README.md
Data Preparation
This repository assumes pre-computed features stored in data/<DATASET>/feats/, including:

DNABERT-S embeddings (+ optional attention priors)

Nucleotide Transformer embeddings (token-level and/or sequence-level)

Structure features:

dsDNA-Model: DNAShape-derived geometry features (e.g., MGW, Roll, ProT, HelT)

ssDNA-Model: RNAfold/RNAplfold-derived secondary-structure statistics (e.g., pairing/unpairing probabilities, energy-related descriptors)

Large intermediate features can be heavy. Consider hosting them via Releases/Zenodo and only keeping a small demo subset in the repo.

Training
Please refer to scripts/ for training entrypoints.
Typical outputs:

checkpoints/dsDNA-Model/<dataset>_foldX_best.pt

checkpoints/ssDNA-Model/<dataset>_foldX_best.pt

threshold files (optional): best_threshold_*.txt, best_delta_*.txt, best_tau_*.txt

Inference (Decision-level Screen–Review)
We recommend the screen–review inference strategy:

Run dsDNA-Model to obtain p_ds

If prediction is uncertain (e.g., |p_ds - 0.5| < δ or u(p_ds) > τ), trigger ssDNA-Model to obtain p_ss

Output final probability using the review rule (e.g., use p_ss for reviewed cases)

See scripts/test_screen_review.py (or your evaluation script) for an implementation.

Evaluation Metrics
We report:

ACC, AUC, MCC, F1

Sensitivity (SN), Specificity (SP)

Review statistics: review rate / review count (for screen–review)

Reproducibility Notes
Fix random seeds where applicable.

Keep the same data splits as in the manuscript.

Record model checkpoints and thresholds for each fold/setting.
