# SSG-Kla

**SSG-Kla: A Sequence-Structure Synergistic Gating Framework with Uncertainty-Aware Fusion for Lysine Lactylation Prediction in Intrinsic Disorder-Enriched Regions**

SSG-Kla is a multimodal framework for predicting lysine lactylation (Kla) sites. It integrates sequence-semantic, evolutionary-context and structural-topology information through three independently trained branches and uncertainty-aware fusion.

This repository provides model implementations, preprocessing scripts, benchmark datasets and partition manifests, external test datasets, and prediction scores from human IDR screening.

## Model overview

![SSG-Kla architecture](assets/Figure1.png)

*Figure 1. Overall architecture of SSG-Kla. Three independent branches capture sequence, evolutionary-context and structural-topology information, followed by uncertainty-aware fusion.*

The model accepts a lysine-centered 51-residue sequence window, with the target lysine at position 26.

- **Sequence-semantic branch:** ESM-2 representations, local convolutional features, window-wide self-attention and bidirectional cross-gating.
- **Evolutionary-context branch:** ProtT5 representations, bidirectional LSTM encoding and residue-wise attention pooling.
- **Structural-topology branch:** SaProt amino-acid–3Di representations, aligned residue features and Cα coordinates, integrated through distance-aware graph attention.

During training, Gaussian coordinate perturbation with a standard deviation of 0.10 Å regularizes distance-based geometric calculations. The 3Di, secondary-structure and accessibility features remain fixed. Validation and inference use unperturbed coordinates.

Uncertainty-aware fusion combines branch logits using sample-specific weights derived from normalized binary predictive entropy and learned global priors. Fusion parameters are fitted using training-set out-of-fold (OOF) predictions. At inference, five-fold probabilities are averaged within each branch before fusion. pLDDT is used for post hoc evaluation rather than as a model input.

## Repository structure

```text
SSG-Kla/
├── README.md
├── environment.yml
├── requirements.txt
├── assets/
│   └── Figure1.png
├── data/
│   ├── benchmark/
│   │   ├── master_D0_manifest.xlsx
│   │   ├── D0_S0/
│   │   │   ├── train_data.xlsx
│   │   │   ├── test_data.xlsx
│   │   │   ├── train_data.npz
│   │   │   └── test_data.npz
│   │   ├── D0_S1/
│   │   │   ├── D0_S1_manifest.xlsx
│   │   │   ├── D0_S1_train_data.xlsx
│   │   │   └── D0_S1_test_data.xlsx
│   │   └── D1_S1/
│   │       ├── D1_S1_manifest.xlsx
│   │       └── D1_S1_train_data.xlsx
│   ├── external_validation/
│   │   ├── human/
│   │   │   └── AttentionKla_human_external_test_769_inference_ready.xlsx
│   │   └── cross_species/
│   │       ├── Rat_test_dataset.xlsx
│   │       ├── Rice_test_dataset.xlsx
│   │       └── Streptococcus_test_dataset.xlsx
│   └── idr/
│       └── DisProt release_2026_06.tsv
├── scripts/
│   ├── preprocessing/
│   │   ├── generate_3di.py
│   │   └── build_structural_npz.py
│   ├── training/
│   │   ├── ESM2.py
│   │   ├── ProtT5.py
│   │   ├── SaProt.py
│   │   └── ensemble_dynamic.py
│   └── prediction/
│       └── test.py
└── results/
    └── idr/
        └── SSG_Kla_idr_predictions.csv
```

Trained checkpoints are distributed separately from the source-code tree.

## Installation

Run the following commands from your chosen project directory:

```bash
git clone https://github.com/flycat200267/SSG-Kla.git
cd SSG-Kla
conda env create -f environment.yml
conda activate ssg-kla
```

Alternatively, install the Python dependencies into an existing compatible Python 3.10 environment:

```bash
python -m pip install -r requirements.txt
```

DSSP is required for structural-feature preparation. If it is not already installed:

```bash
conda install -c conda-forge dssp
```

GPU execution requires a compatible PyTorch build and NVIDIA driver.

## Datasets

### Benchmark datasets

D0 contains 17,053 human lysine sites: 8,526 Kla-positive and 8,527 label-negative sites. Label-negative sites are lysines without recorded Kla annotations in the source data.

D0 and D1 describe sample construction; S0 and S1 describe partitioning.

| Setting | Partition | Kla-positive | Label-negative | Total |
| --- | --- | ---: | ---: | ---: |
| D0+S0 | Training | 6,821 | 6,821 | 13,642 |
| D0+S0 | Test | 1,705 | 1,706 | 3,411 |
| D0+S1 | Training | 6,821 | 6,820 | 13,641 |
| D0+S1 | Test | 1,705 | 1,707 | 3,412 |
| D1+S1 | Training | 6,821 | 6,820 | 13,641 |
| D1+S1 | Test, shared with D0+S1 | 1,705 | 1,707 | 3,412 |

- **D0+S0:** Primary benchmark using an approximately 8:2 site-level split.
- **D0+S1:** The D0 site collection under a homology-cluster-disjoint partition. The manifest records the partition and predefined cross-validation folds.
- **D1+S1:** Alternative training negatives, retaining the S1 training positives and test set.

D1+S1 uses `data/benchmark/D0_S1/D0_S1_test_data.xlsx` for testing.

The supplied `train_data.npz` and `test_data.npz` correspond to D0+S0. Other datasets require their own aligned structural archives.

### External test datasets

The independent human test dataset contains 769 sites from 495 proteins. It was derived from AttentionKla test data after sequence verification, protein-overlap removal and homology filtering against D0.

Cross-species datasets contain rat, rice and *Streptococcus mutans* samples derived from public HybridKla data. They were evaluated using the human-trained model without species-specific retraining.

## Input formats

### Site tables

| Column | Description |
| --- | --- |
| `UniProt ID` | Protein accession; prediction also accepts `UniProt_ID` |
| `Position` | One-based target position in the source protein |
| `Sequence` | Lysine-centered 51-residue window |
| `foldseek_seq` | Residue-aligned 3Di encoding |
| `Label` | 1 for Kla-positive and 0 for label-negative benchmark sites |

Training requires labels. Prediction supports unlabeled inputs.

Windows extending beyond protein termini use X padding. The provided preprocessing workflow generates sequence-derived 3Di states using ProstT5; `foldseek_seq` is retained as the compatible column name.

### Structural archives

| NPZ field | Shape or value | Description |
| --- | --- | --- |
| `coords` | `[N, 51, 3]` | Cα coordinates in Å |
| `aa_ids` | `[N, 51]` | Amino-acid encodings |
| `ss_ids` | `[N, 51]` | Secondary-structure encodings |
| `rsa_vals` | `[N, 51, 1]` | Relative solvent accessibility |
| `valid_mask` | `[N, 51]` | Valid-coordinate mask |
| `dssp_mask` | `[N, 51]` | Valid-DSSP-feature mask |
| `residue_ids` | `[N, 51]` | Protein residue positions |
| `uniprot_ids` | `[N]` | Protein accessions |
| `target_positions` | `[N]` | Target lysine positions |
| `center_index` | `25` | Zero-based target index |
| `window_size` | `51` | Window length |

The updated structural builder additionally exports `sample_keys` and `metadata`.

**The site table and NPZ archive must contain the same samples in the same row order**, including any filtering of training labels.

Missing coordinates are handled using validity masks. For samples without valid coordinates, the structural branch uses its sequence-derived pooled representation as the fallback.

## Pretrained backbones and trained checkpoints

The scripts require local pretrained backbones and matching tokenizers:

| Component | Required model |
| --- | --- |
| Sequence-semantic branch | ESM-2 `esm2_t30_150M_UR50D` |
| Evolutionary-context branch | ProtT5 matching the training configuration |
| Structural-topology branch | SaProt matching the trained checkpoints |
| 3Di generation | ProstT5 and the tokenizer used by `generate_3di.py` |

Trained SSG-Kla checkpoints and fitted fusion parameters are planned for distribution through [GitHub Releases](https://github.com/flycat200267/SSG-Kla/releases). The release will contain five fold-specific checkpoints for each branch and instructions for extraction and loading.

| Branch | Fold-specific model names |
| --- | --- |
| ESM-2 | `ESM2_branch_fold_1` through `ESM2_branch_fold_5` |
| ProtT5 | `ProtT5_branch_fold_1` through `ProtT5_branch_fold_5` |
| SaProt | `SaProt_branch_fold_1.pth` through `SaProt_branch_fold_5.pth` |

ESM-2 and ProtT5 models are stored as directories. Preserve the complete exported directories, including the configuration and classifier files required for loading.

For split archives, download all parts and follow the reconstruction instructions accompanying the release.

## Prediction using trained checkpoints

Paths and settings are configured inside the scripts; command-line arguments are not required. Use absolute local paths for input, model and output settings to avoid ambiguity. Run the commands below from the repository root.

After the checkpoints are released:

1. Download and extract the branch checkpoints and fusion configuration.
2. Prepare a site table and its aligned structural NPZ archive.
3. Configure `scripts/prediction/test.py` with the input files, pretrained backbones, checkpoint directories and output location.
4. Copy the fitted fusion parameters into the corresponding prediction variables.

Checkpoint-directory settings must point to the parent directories containing the five fold models:

- `ESM2_CHECKPOINT_DIR`
- `PROTT5_CHECKPOINT_DIR`
- `SAPROT_CHECKPOINT_DIR`

| Fusion JSON key | Prediction variable |
| --- | --- |
| `theta_esm2` | `THETA_ESM2` |
| `theta_prott5` | `THETA_PROTT5` |
| `theta_saprot` | `THETA_SAPROT` |
| `lambda` | `LAMBDA` |

Run prediction:

```bash
python scripts/prediction/test.py
```

The script averages probabilities across five checkpoints within each branch, applies uncertainty-aware fusion and saves predictions to `OUTPUT_PATH`.

## Training from scratch

### Default training configuration

| Setting | ESM-2 | ProtT5 | SaProt |
| --- | --- | --- | --- |
| Optimizer | AdamW | AdamW | AdamW |
| Learning rate | 4e-5 | 1e-4 | 1e-4 |
| Batch size | 16 | 8 | 16 |
| Gradient accumulation | 1 | 4 | 4 |
| Effective batch size | 16 | 32 | 64 |
| Maximum epochs | 30 | 30 | 25 |
| Weight decay | 0.01 | 0.01 | 0.15 |
| Early-stopping patience | 6 | 6 | 8 |
| Learning-rate schedule | Cosine | Cosine | Linear |
| Warm-up ratio | 0.1 | 0.1 | 0.1 |
| Cross-validation folds | 5 | 5 | 5 |
| Backbone adaptation | Full fine-tuning | LoRA r=16, α=32 | LoRA r=64, α=128 |

LoRA dropout is 0.1. Checkpoints are selected using validation AUC.

### Train the branches

Configure the dataset, pretrained model and output paths at the top of each training script. For D0+S0, use `data/benchmark/D0_S0/train_data.xlsx` and the corresponding `train_data.npz` for SaProt.

```bash
python scripts/training/ESM2.py
python scripts/training/ProtT5.py
python scripts/training/SaProt.py
```

The default scripts use stratified five-fold cross-validation with shuffling and random state 42. Each branch saves fold-specific checkpoints and OOF predictions.

For D0+S1 or D1+S1, use the predefined `CV_Fold` assignments in the corresponding manifest instead of the default `StratifiedKFold`. Changing only the input-table path does not apply the S1 fold protocol.

### Fit fusion parameters

Configure `FILE_ESM`, `FILE_T5` and `FILE_SAPROT` in `ensemble_dynamic.py` to point to the branch OOF files:

```bash
python scripts/training/ensemble_dynamic.py
```

The OOF inputs must contain `label` and `score` columns and represent the same ordered training sites.

Fusion fitting uses training OOF predictions. The fitted parameters can then be used for prediction without adjustment on the test data.

## Preparing additional inputs

The supplied D0+S0 spreadsheets already contain 3Di encodings, and their structural archives are provided.

For new datasets, configure and run the preprocessing scripts as needed:

```bash
python scripts/preprocessing/generate_3di.py
python scripts/preprocessing/build_structural_npz.py
```

- `generate_3di.py` generates 3Di encodings using ProstT5.
- `build_structural_npz.py` prepares aligned coordinates, residue features and validity masks from AlphaFold structures.

The structural builder uses cached PDB files when available and downloads missing files. DSSP operates on a temporary copy with DBREF records removed; the original PDB files are preserved.

## Prediction outputs

| Output field | Description |
| --- | --- |
| `UniProt_ID`, `Position`, `Sequence` | Site identifiers and input window |
| `ESM2_Prob`, `ProtT5_Prob`, `SaProt_Prob` | Five-fold mean branch probabilities |
| `ESM2_Entropy`, `ProtT5_Entropy`, `SaProt_Entropy` | Normalized predictive entropies |
| `ESM2_Weight`, `ProtT5_Weight`, `SaProt_Weight` | Sample-specific fusion weights |
| `Final_Prob` | Final fused probability |
| `Prediction` | Binary prediction at a cutoff of 0.5 |
| `foldseek_fallback` | Indicates replacement of an invalid-length 3Di string |
| `valid_structure_residues` | Number of valid coordinate positions |
| `Label` | Included when valid labels are supplied |

Evaluation metrics are also reported when suitable labels are available.

The **0.5 classification cutoff** is separate from the **0.85 IDR candidate-prioritization threshold**.

## Reported performance

| Evaluation | AUC | AP | ACC | MCC |
| --- | ---: | ---: | ---: | ---: |
| Primary D0+S0 test set | 0.8755 | 0.8712 | 0.7883 | 0.5816 |
| D0+S1 | 0.8641 | 0.8519 | 0.7855 | 0.5732 |
| D1+S1 | 0.8538 | 0.8352 | 0.7717 | 0.5457 |
| Independent human test set | 0.8456 | 0.8183 | 0.7711 | 0.5470 |
| IDR-enriched pLDDT <50 subset | 0.9061 | 0.9102 | 0.8322 | 0.6620 |

Full comparisons, confidence intervals and analytical procedures are provided in the manuscript and Supplementary Information. Low pLDDT indicates low structural confidence and is not equivalent to experimentally annotated disorder.

## Human IDR screening

The DisProt source annotations are provided in:

```text
data/idr/DisProt release_2026_06.tsv
```

Predictions for 9,692 screened lysine-centered windows are provided in:

```text
results/idr/SSG_Kla_idr_predictions.csv
```

Applying `Final_Prob >= 0.85` identifies 1,276 prioritized candidates:

```python
import pandas as pd

scores = pd.read_csv("results/idr/SSG_Kla_idr_predictions.csv")
candidates = scores.loc[scores["Final_Prob"] >= 0.85].copy()
```

These are model-prioritized candidates for experimental follow-up. Labels retained in the screening table should not be interpreted as experimentally established Kla labels.

The associated phosphosite-proximity analysis identified 207 candidates within ±5 residues of annotated phosphosites, compared with 187.7513 expected under protein-matched permutation, yielding 1.1025-fold enrichment. Threshold sensitivity and functional-type analyses are described in the Supplementary Information.

## Citation

If you use SSG-Kla or its resources, please cite the accompanying manuscript:

*SSG-Kla: A Sequence-Structure Synergistic Gating Framework with Uncertainty-Aware Fusion for Lysine Lactylation Prediction in Intrinsic Disorder-Enriched Regions.*

The final bibliographic information will be added after publication. Please also cite the original studies for reused datasets and pretrained models.

## Contact

For questions about the code or datasets, please [open an issue](https://github.com/flycat200267/SSG-Kla/issues), including the relevant script name and a description of the problem.
