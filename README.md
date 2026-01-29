# MLMGCN-CVD
LLM-driven multimodal semantic graph learning framework for gut microbiome metagenomic DNA sequence-based ACVD risk prediction

## Highlights
- Multimodal node embeddings: DNABERT-S (semantic) + Nucleotide Transformer (context) + structure channel
- Two sub-models:
  - dsDNA-Model: structure channel uses DNAShape (dsDNA geometry)
  - ssDNA-Model: structure channel uses RNAfold/RNAplfold (ssDNA secondary-structure statistics)
- Decision-level screen–review strategy: dsDNA screening, ssDNA review on uncertain samples

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

Training/testing scripts read `labels.csv` containing:

- `label` column (0/1)

Example:

```csv
label
1
0
```

**Important:** The row order in `labels.csv` must match the sequence order used for feature extraction, since sample ids are generated as `000000`, `000001`, ...

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

### 3) Training

Edit in `code/Model_Train.py`:

- `FEAT_ROOT`  
- `LABELS_CSV`  
- hyperparameters (EPOCHS, LR, etc.)

Run:

```bash
python code/Model_Train.py
```

Checkpoints will be saved in `checkpoints/` (or the output directory configured in your script).

---

### 4) Testing / Evaluation

Edit in `code/Model_Test.py`:

- `FEAT_ROOT`  
- `LABELS_CSV`  
- `SAVE_DIR`  

Run:

```bash
python code/Model_Test.py
```

---

## Notes

- Models will be downloaded from HuggingFace when needed (network required), unless you set local paths.  
- For reproducibility, fix random seeds and keep consistent data ordering.  
- Large files (raw datasets, cached embeddings, checkpoints) are recommended to be hosted outside GitHub and linked in this README.

---

## Acknowledgement

If you use this repository in academic work, please cite your corresponding paper.

---

