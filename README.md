# SSG-Kla: A Sequence-Structure Synergistic Gating Framework with Multi-Modal Decoupling and Uncertainty-Aware Fusion for Lysine Lactylation Prediction

## Introduction

Lysine Lactylation (Kla) plays a crucial role in metabolic regulation and epigenetic remodeling. 

SSG-Kla is a novel Sequence-Structure Synergistic Gating framework designed to address cross-modal interference and generalization bottlenecks in PTM prediction. 

By constructing three parallel decoupled branches, SSG-Kla integrates:
* A sequence semantic module utilizing ESM-2 and a bidirectional cross-gating mechanism.
* An evolutionary context module based on ProtT5 and BiLSTM.
* A spatial topological module integrating SaProt with an Equivariant Graph Attention Network (EGAT).

To mitigate structural uncertainties, the model introduces a dynamic late fusion mechanism based on information entropy , achieving highly accurate prediction of Kla sites.

<img width="3456" height="3253" alt="A_graph_" src="https://github.com/user-attachments/assets/8518b9b6-fcfa-4c59-b40d-1cca70e528b8" />

## Installation

1. Clone the repository

   ```
   git clone [https://github.com/flycat200267/SSG-Kla.git](https://github.com/flycat200267/SSG-Kla.git)
   cd SSG-Kla
   ```

2. Create and activate a Conda environment

   ```
   conda create -n ssgkla python=3.9
   conda activate ssgkla
   ```

3. Install dependencies

   ```
   pip install torch torchvision torchaudio
   pip install transformers
   pip install -r requirements.txt
   ```
## Repository Structure

The repository is organized as follows, aligning with the multi-modal decoupling and uncertainty-aware fusion architecture of SSG-Kla:

```
SSG-Kla/
├── data/                      # Dataset directory
│   ├── source.xlsx            # The original complete dataset
│   ├── train_data.xlsx        # Benchmark training set for model fitting
│   └── test_data.xlsx         # Independent test set for evaluating generalization
├── train/                     # Multi-modal feature learning and fusion modules
│   ├── ESM2.py                # Sequence semantic module (ESM-2 with cross-gating)
│   ├── ProtT5.py              # Evolutionary context module (ProtT5 with BiLSTM)
│   ├── SaProt.py              # Spatial topological module (SaProt with EGAT)
│   └── ensemble_dynamic.py    # Uncertainty-aware dynamic late fusion mechanism
└── test/                      # Evaluation and prediction module
    └── test.py                # Evaluation script to compute performance metrics on the test set
```

