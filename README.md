# MLMGCN-CVD
A multimodal graph learning framework for sequence-level ACVD risk prediction from gut microbiome metagenomic DNA fragments.

## Highlights
- Multimodal node embeddings: DNABERT-S (semantic) + Nucleotide Transformer (context) + structure channel
- Content-aware semantic graph construction based on fused token-level representations
- GCN-CNN cooperative backbone for graph propagation and local motif/structural refinement
- Optional readmap-derived abundance and copy-number features integrated through a late-fusion projection-and-gating branch
- Two sub-models:
  - dsDNA-Model: structure channel uses DNAshapeR (dsDNA geometry)
  - ssDNA-Model: structure channel uses RNAfold/RNAplfold (ssDNA secondary-structure statistics)
- Decision-level screen-review strategy: dsDNA screening, ssDNA review on uncertain samples

## Get the Source

```bash
git clone https://github.com/xing1999/MLMGCN-CVD.git
cd MLMGCN-CVD
```

---

## Required Dependencies

- Python >= 3.9  
- numpy  
- pandas  
- scikit-learn  
- tqdm  
- PyTorch  
- transformers  

Install:

```bash
pip install -r requirements.txt
```

---

## Optional Dependencies (for structural / DNAshape features)

### (A) DNAshapeR (R)

This project can optionally extract DNAshape features via **R + DNAshapeR**.

1. Install **R (>= 4.0)**
2. Install **DNAshapeR** in R:

```r
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install("DNAshapeR")
```

### (B) ViennaRNA (RNAfold / RNAplfold)

If you use `code/SS_StructFeature.py` for secondary-structure style features, install **ViennaRNA** and ensure `RNAfold` / `RNAplfold` are available in your system `PATH`.

---

## Data Source (ENA)

Raw metagenomic shotgun sequencing reads are publicly available from the European Nucleotide Archive (ENA) under study accession **ERP023788**:

```text
https://www.ebi.ac.uk/ena/browser/view/ERP023788?show=related-records
```

### Programmatic download (recommended)

You can retrieve run accessions and FASTQ download paths using the ENA Portal API `filereport`:

```bash
# generate a TSV containing FTP links and MD5 checksums for all runs
curl -L "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=ERP023788&result=read_run&fields=run_accession,fastq_ftp,fastq_md5,fastq_bytes&format=tsv" \
  -o data/ERP023788_fastq.tsv
```

From `data/ERP023788_fastq.tsv`, you can download the FASTQ files using tools like `wget`, `curl`, `aria2c`, ENA File Downloader, Globus, or Aspera (choose based on your environment and network).

> Note: ENA provides both submitted files and archive-generated files for runs. Use the file paths and MD5 in the report to verify download integrity.

---

## Data Format

### Sequence CSV

Feature extraction expects a CSV file containing at least:

- `sequence` column (DNA sequence, A/C/G/T)

Example:

```csv
sequence
ACGT...
TTAG...
```

### Labels CSV

Training/testing scripts read `labels.csv` containing at least:

- `label` column (0/1)

Example:

```csv
label
1
0
```

**Important:** The row order in `labels.csv` must match the sequence order used for feature extraction, since sample ids are generated as `000000`, `000001`, ...

### Optional readmap-derived abundance and copy-number features

MLMGCN-CVD can additionally use readmap-derived quantitative features as auxiliary global inputs. These features are not used for node construction or semantic graph construction. Instead, they are processed by a quantitative projection-and-gating branch and fused with the graph-derived sequence representation before final classification.

When available, the following 17 readmap-derived abundance and copy-number columns can be included in `labels.csv` or the metadata CSV used by the training/testing scripts:

- `readmap_mapped_reads`
- `readmap_covbases`
- `readmap_breadth`
- `readmap_mean_depth`
- `readmap_depth_per_million_reads`
- `readmap_cpm`
- `readmap_rpkm`
- `readmap_tpm`
- `readmap_copy_number_mean_depth`
- `readmap_copy_number_median_depth`
- `readmap_copy_number_log2_mean_depth`
- `readmap_copy_number_log2_median_depth`
- `readmap_mapped_fragments`
- `readmap_fragment_mean_depth`
- `readmap_fpkm`
- `readmap_copy_number_fragment_mean_depth`
- `readmap_copy_number_log2_fragment_mean_depth`

Example:

```csv
label,readmap_mapped_reads,readmap_covbases,readmap_breadth,readmap_mean_depth,readmap_tpm
1,148,962,0.962,3.21,15.7
0,32,410,0.410,0.84,3.2
```

**Important:** The row order of readmap-derived quantitative features must match the sequence features and labels.

---

## Usage

### 1) Feature Extraction (DNABERT-S / Nucleotide Transformer / DNAshape)

Edit paths in `code/MLLMFeature.py` (typical variables; names may differ slightly in your script):

- `CSV_PATH` : input CSV with `sequence` column  
- `OUT_ROOT` : output feature folder (e.g., `data/feats`)  
- `USE_DNABERTS` : DNABERT-S model path or model id  
- `NT_MODEL_ID` : default `InstaDeepAI/nucleotide-transformer-500m-1000g`  
- `RSCRIPT_EXE` : path to `Rscript.exe` (if DNAshape enabled)

Run:

```bash
python code/MLLMFeature.py
```

It will create folders like:

- `dnaberts_emb/`
- `dnaberts_attn/`
- `nt_emb_tok/`
- `nt_emb_seq/`
- `dnashape/` (if enabled)

---

### 2) (Optional) SS Structure Feature Extraction

If you want to build `ss_struct` features:

```bash
python code/SS_StructFeature.py --csv data/data.csv --outdir data/feats/ss_struct
```

> Adjust `--csv` to your actual file. By default, CSV should contain `sequence` and `label` columns.

---

### 3) (Optional) Readmap-Derived Quantitative Features

If read-mapping results are available, append the 17 readmap-derived abundance and copy-number columns to `labels.csv` or the metadata CSV used by the training/testing scripts.

These features provide fragment-level quantitative context, including mapped reads, coverage breadth, sequencing depth, normalized abundance indices, and depth-based copy-number estimates. They are treated as global scalar features and are introduced only at the late-fusion stage.

The quantitative branch contains:

- `quant_proj`: projects abundance/copy-number features into the hidden representation space
- `quant_gate`: generates a gate vector to modulate the graph-derived sequence representation

The final fused representation is constructed as:

```text
[g * (1 + gate + q_gate), z_seq, z_quant]
```

where `g` is the graph-derived sequence representation, `z_seq` is the projected global Nucleotide Transformer representation, and `z_quant` is the projected readmap-derived quantitative representation.

---

### 4) Training

Edit in `code/Model_Train.py`:

- `FEAT_ROOT`  
- `LABELS_CSV`  
- hyperparameters (EPOCHS, LR, etc.)
- quantitative feature settings if readmap-derived features are used

Run:

```bash
python code/Model_Train.py
```

The model reads token-level sequence features from `FEAT_ROOT` and labels from `LABELS_CSV`. If readmap-derived quantitative columns are present, they are used as auxiliary global features in the late-fusion branch.

Checkpoints will be saved in `checkpoints/` (or the output directory configured in your script).

---

### 5) Testing / Evaluation

Edit in `code/Model_Test.py`:

- `FEAT_ROOT`  
- `LABELS_CSV`  
- `SAVE_DIR`  

Run:

```bash
python code/Model_Test.py
```

For evaluation with readmap-derived quantitative features, make sure that the testing metadata CSV contains the same readmap feature columns used during training and that the row order matches the extracted sequence features.

---

## Model Overview

MLMGCN-CVD uses fixed-length gut metagenomic DNA fragments as input. DNABERT-S, Nucleotide Transformer, and sequence-derived structural features are aligned and fused into node-level representations. A content-aware semantic graph is constructed from these token-level representations.

The semantic graph is processed by a GCN-CNN cooperative backbone. Graph convolution captures long-range topological dependencies among sequence segments, while 1D convolutional refinement captures local motif and structural patterns. A multi-head attention readout aggregates node-level features into a fragment-level graph representation.

If readmap-derived abundance and copy-number features are provided, they are processed as auxiliary global quantitative features through a projection-and-gating branch. They are fused with the graph-derived representation only at the final late-fusion stage.

---

## Notes

- The semantic graph is constructed from token-level sequence-derived features, including DNABERT-S, Nucleotide Transformer, and structural features.  
- Readmap-derived abundance and copy-number features are optional auxiliary global inputs.  
- Readmap-derived quantitative features are not used to construct sequence nodes or graph edges.  
- The quantitative branch uses late fusion rather than a fixed post hoc probability correction.  
- The model does not apply a manually capped abundance correction or a fixed probability shift.  
- For reproducibility, fix random seeds and keep consistent row ordering across sequence files, feature folders, labels, and readmap-derived quantitative features.  
- Models will be downloaded from HuggingFace when needed (network required), unless you set local paths.  
- Large files (raw datasets, cached embeddings, readmap tables, checkpoints) are recommended to be hosted outside GitHub and linked in this README.

---

## Acknowledgement

If you use this repository in academic work, please cite your corresponding paper.

---
