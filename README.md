# Coupling Local Context and Global Semantic Prototypes via a Hierarchical Architecture for Rhetorical Roles Labeling

<p align="center">
  <a href="https://aclanthology.org/2026.eacl-long.137/"><img src="https://img.shields.io/badge/Paper-EACL%202026-blue?style=flat-square&logo=read-the-docs" alt="Paper"></a>
  <a href="https://creativecommons.org/licenses/by/4.0/"><img src="https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey?style=flat-square" alt="License"></a>
  <img src="https://img.shields.io/badge/Conference-EACL%202026-orange?style=flat-square" alt="Conference">
  <img src="https://img.shields.io/badge/Python-3.9%2B-yellow?style=flat-square&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-Framework-orange?style=flat-square&logo=pytorch" alt="PyTorch">
</p>

<p align="center">
  <b>Anas Belfathi¹ &nbsp;·&nbsp; Nicolas Hernandez¹ &nbsp;·&nbsp; Laura Monceaux¹ &nbsp;·&nbsp; Warren Bonnard²</b><br>
  <b>Mary Catherine Lavissière¹ &nbsp;·&nbsp; Christine Jacquin¹ &nbsp;·&nbsp; Richard Dufour¹</b><br>
  <i>¹ Nantes Université, École Centrale Nantes, CNRS, LS2N, UMR 6004, F-44000 Nantes, France</i><br>
  <i>² University of Lorraine, France</i>
</p>

---

## Abstract

Rhetorical Role Labeling (RRL) identifies the functional role of each sentence in a document, a key task for discourse understanding in domains such as law and medicine. While hierarchical models capture local dependencies effectively, they are limited in modeling global, corpus-level features.

We propose two prototype-based methods that integrate local context with global representations:

- **Prototype-Based Regularization (PBR)** — learns soft prototypes through a distance-based auxiliary loss to structure the latent space without altering the backbone architecture.
- **Prototype-Conditioned Modulation (PCM)** — constructs corpus-level prototypes and injects them into the hierarchical encoder during both training and inference.

We also introduce **SCOTUS-LAW**, the first dataset of U.S. Supreme Court opinions annotated with rhetorical roles at three levels of granularity: *category*, *rhetorical function*, and *step*. Experiments on legal, medical, and scientific benchmarks show consistent improvements over strong baselines, with **~4 Macro-F1 gains on low-frequency roles**.

---

## Repository Structure

```
RRL-HierProto/
├── datasets/                          # Dataset loading utilities (data not included, see below)
├── context-extraction/                # Context retrieval heuristics
├── document-grouping/                 # Supervised document clustering for PCM sampling
├── matching-context/                  # Prototype matching and injection utilities
├── slurms/                            # SLURM scripts for HPC (IDRIS/JeanZay)
│
├── models.py                          # PBR and PCM model architectures
├── train.py                           # Main training pipeline
├── baseline_run.py                    # HSLN baseline (no prototypes)
├── eval.py / eval_run.py              # Evaluation scripts
├── dataset_reader.py                  # Dataset loading and preprocessing
├── tokenize_files.py                  # Tokenization utilities
├── data_prep.py                       # Data preparation pipeline
├── batch_creator.py                   # Batch construction for training
├── context_fusion.py                  # Prototype injection strategies
├── compute_metrics.py                 # Metrics computation (mF1, wF1)
├── organize_embeddings.py             # Prototype extraction and storage
├── SummaryGeneration.py               # Summary generation utilities
├── requirements.txt                   # Python dependencies
└── README.md
```

> ⚠️ **Note:** The `datasets/` folder contains loading utilities only. The annotated SCOTUS-LAW data is not included in this repository for privacy reasons. See the [Dataset](#dataset) section for access instructions.

---

## Methods

### Backbone: Hierarchical Sequential Labeling Network (HSLN)

All experiments build on the state-of-the-art HSLN architecture. Each sentence is encoded independently with BERT, then passed through a Bi-LSTM and attention-pooling to obtain fixed-size sentence vectors. A second Bi-LSTM contextualizes these vectors with surrounding sentences, and a CRF layer predicts the optimal label sequence.

### Prototype-Based Regularization (PBR)

PBR enriches the hierarchical architecture with **trainable soft prototypes** that share the embedding space with sentence vectors. An auxiliary loss steers representations toward corpus-level rhetorical patterns without modifying the backbone:

```
L = L_task + λ_prox · L_prox − λ_div · L_div
```

- **L_prox** — pulls sentence embeddings toward their nearest prototype
- **L_div** — encourages prototypes to spread out, reducing redundancy

### Prototype-Conditioned Modulation (PCM)

PCM **precomputes role prototypes** from the training corpus and injects them into the hierarchical encoder through conditioning modules. The process involves three stages:

1. **Document sampling** — Full, Random, or Supervised (K-Means clustering)
2. **Prototype extraction** — Average embeddings of sentences per rhetorical role
3. **Prototype injection** — Cosine similarity-based assignment + modulation (Linear Fusion, CLN, Gated Residual, FiLM, Cross-Attention)

---

## Dataset

### SCOTUS-LAW

We introduce **SCOTUS-LAW**, the first publicly available corpus of U.S. Supreme Court opinions annotated with rhetorical roles at three levels of granularity.

| Split | Documents | Sentences | Avg. Sentences/Doc |
|---|---|---|---|
| Train | 144 | 21,396 | 148.58 |
| Dev | 18 | 2,450 | 136.11 |
| Test | 18 | 2,481 | 137.83 |
| **Total** | **180** | **26,327** | — |

The annotation scheme operates at three levels:

```
Step = Discursive Category + Rhetorical Function + Optional Attributes
```

**5 Discursive Categories:** Setting the scene · Analysis · Resolution · Sources of authority · Announcing

**13 Rhetorical Functions:** Recalling · Quoting · Presenting jurisdiction · Stating the Court's reasoning · Describing · Giving the holding · Citing · Rejecting arguments · Announcing · Granting certiorari · Giving instructions · Accepting arguments · Evaluating impact

> 📧 **Data Access:** The SCOTUS-LAW dataset is not included in this repository for privacy reasons. Please contact the authors at `anas.belfathi@univ-nantes.fr` to request access.

### Other Evaluation Datasets

| Dataset | Domain | Labels | # Docs | # Sentences |
|---|---|---|---|---|
| SCOTUSCategory | Legal (U.S.) | 5 | 180 | 26,327 |
| SCOTUSRF | Legal (U.S.) | 13 | 180 | 26,327 |
| SCOTUSSteps | Legal (U.S.) | 35 | 180 | 26,327 |
| LegalEval | Legal (India) | 13 | 214 | 31,865 |
| DeepRhole | Legal (India) | 7 | 50 | 9,380 |
| PubMed | Medical | 5 | 20,000 | 227,000 |
| CS-Abstracts | Scientific | 5 | 654 | 7,385 |

---

## Installation

```bash
git clone https://github.com/AnasBelfathi/RRL-HierProto.git
cd RRL-HierProto
pip install -r requirements.txt
```

---

## Usage

### 1. Data Preparation

```bash
python data_prep.py --dataset legaleval --data_path datasets/legaleval/
python tokenize_files.py --dataset legaleval --model bert-base-uncased
```

### 2. Prototype Extraction (PCM only)

```bash
# Extract and organize sentence embeddings for prototype computation
python organize_embeddings.py \
    --dataset scotus_rf \
    --model legal-bert-base-uncased \
    --sampling full
```

### 3. Run Baseline (HSLN)

```bash
python baseline_run.py \
    --train datasets/legaleval/train.json \
    --dev datasets/legaleval/dev.json \
    --test datasets/legaleval/test.json \
    --model bert-base-uncased \
    --epochs 40
```

### 4. Run PBR

```bash
python train.py \
    --dataset legaleval \
    --model bert-base-uncased \
    --method pbr \
    --lambda_prox 0.9 \
    --lambda_div 0.9 \
    --num_prototypes 16 \
    --epochs 40
```

### 5. Run PCM

```bash
python train.py \
    --dataset scotus_rf \
    --model legal-bert-base-uncased \
    --method pcm \
    --centroids_path centroids_subset_rf.joblib \
    --injection linear_fusion \
    --sampling full \
    --epochs 40
```

### 6. Evaluate

```bash
python eval_run.py \
    --model_path checkpoints/best_model.pt \
    --test datasets/legaleval/test.json
```

### HPC (SLURM/JeanZay)

```bash
cd slurms/
sbatch run_pbr_scotus.sh
sbatch run_pcm_legaleval.sh
```

---

## Results

Performance (Macro-F1 / Weighted-F1) across legal, medical, and scientific benchmarks. † and ‡ indicate statistical significance over the baseline at p=0.05 and p=0.01.

| Model | SCOTUSCat | SCOTUSRF | SCOTUSSteps | LegalEval | DeepRhole | PubMed | CS-Abstracts |
|---|---|---|---|---|---|---|---|
| | mF1 / wF1 | mF1 / wF1 | mF1 / wF1 | mF1 / wF1 | mF1 / wF1 | mF1 / wF1 | mF1 / wF1 |
| Baseline (HSLN) | 82.22 / 88.35 | 61.36 / 78.81 | 46.70 / 63.21 | 78.82 / 90.94 | 44.24 / 50.51 | 87.01 / 91.09 | 68.55 / 75.08 |
| Mind (T.y.s.s. 2024) | 83.46 / 89.20 | 62.67 / 79.07 | 45.24 / 62.78 | 79.80 / 91.25 | 45.30 / 50.93 | 87.67 / 91.86 | 69.19 / 76.91 |
| **PBR** | **83.69‡ / 89.75‡** | **65.75‡ / 80.31‡** | **50.48‡ / 65.73‡** | **82.50‡ / 93.17‡** | 44.96† / 51.11† | **88.86‡ / 92.91‡** | **71.10‡ / 78.09‡** |
| PCM (Full) | 83.96‡ / 89.80‡ | 67.53‡ / 80.64‡ | 54.03‡ / 67.54‡ | 81.41‡ / 91.21 | **47.13‡ / 55.54‡** | 87.19 / 91.89 | 69.84 / 76.66 |
| PCM (Random) | 83.93‡ / 89.70‡ | 67.24‡ / 80.66‡ | 54.62‡ / 67.55‡ | 81.83‡ / 91.57 | 47.30‡ / 53.90‡ | 87.24 / 91.94 | 69.12 / 76.30† |
| **PCM (Supervised)** | **84.13‡ / 89.75‡** | **67.45‡ / 80.92‡** | **54.40‡ / 67.79‡** | 80.77‡ / 91.00 | 45.92‡ / 53.86‡ | 87.42 / 92.06† | 68.69 / 75.46 |

### vs. LLMs Fine-Tuned with QLoRA (on SCOTUSRF)

| Model | mF1 | wF1 |
|---|---|---|
| DeepSeek-70B | 65.20 | 75.20 |
| Meta-Llama3-8B | 66.78 | 75.09 |
| Mistral-7B | 70.29 | 76.61 |
| Qwen3-8B | 69.36 | 75.53 |
| **PCM (Ours)** | 65.75 | **80.31** |
| **PBR (Ours)** | 67.45 | **80.92** |

Our prototype-based methods surpass fine-tuned LLMs on weighted-F1 **with ~70× fewer parameters**.

### Key Findings

- **PBR** provides consistent mF1 gains across all 7 benchmarks with minimal overhead (20–25% faster training, 30–40% less GPU memory than PCM).
- **PCM** achieves the best results on 4 of 7 tasks, most notably on SCOTUSSteps (+7.3 mF1 over baseline).
- Both methods are **particularly effective on low-frequency and ambiguous roles** (e.g., +41.75 F1 on Accepting arguments/a reasoning on SCOTUSRF).
- **Supervised sampling** helps on broad label sets; all strategies converge on simpler corpora (PubMed, CS-Abstracts).

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Base model | `bert-base-uncased` |
| LSTM dimension | 768 |
| Attention context dim | 200 |
| Dropout | 0.5 |
| Max sequence length | 128 |
| Learning rate | {1e-5, 3e-5, 5e-5, 1e-4, 3e-4} |
| Epochs | 40 |
| Optimizer | Adam |
| PBR prototypes (Q) | {2, 4, 8, 16, 32, 64} |
| λ_prox, λ_div | {0, 0.9, 10} |

---

## Related Work

This repo is part of a broader research line on rhetorical role labeling:

> Belfathi, A., Hernandez, N., Monceaux, L. (2023). *Harnessing GPT-3.5-turbo for Rhetorical Role Prediction in Legal Cases.* JURIX 2023. [[Paper]](https://hal.science/hal-04264675) [[Code]](https://github.com/AnasBelfathi/In-Context-Learning-RRL)

> Belfathi, A., Hernandez, N., Monceaux, L., Dufour, R. (2025). *A Simple but Effective Context Retrieval for Sequential Sentence Classification in Long Legal Documents.* ArgMining @ ACL 2025. [[Paper]](https://aclanthology.org/2025.argmining-1.15/) [[Code]](https://github.com/AnasBelfathi/ContextRRL)

> Belfathi, A., Gallina, Y., Hernandez, N., Monceaux, L., Dufour, R. (2025). *Is Selective Masking A Key to Improving Domain Adaptation for Masked Language Model?* ICAIL 2025. [[Paper]](https://doi.org/10.1145/3769126.3769216) [[Code]](https://github.com/ygorg/legal-masking)

---

## Citation

If you use this code or find our work useful, please cite:

```bibtex
@inproceedings{belfathi-etal-2026-coupling,
    title     = "Coupling Local Context and Global Semantic Prototypes via a Hierarchical 
                 Architecture for Rhetorical Roles Labeling",
    author    = "Belfathi, Anas and Hernandez, Nicolas and Laura, Monceaux and
                 Bonnard, Warren and Lavissière, Mary Catherine and
                 Jacquin, Christine and Dufour, Richard",
    booktitle = "Proceedings of the 19th Conference of the European Chapter of the 
                 Association for Computational Linguistics (Volume 1: Long Papers)",
    month     = mar,
    year      = "2026",
    address   = "Rabat, Morocco",
    publisher = "Association for Computational Linguistics",
    url       = "https://aclanthology.org/2026.eacl-long.137/",
    doi       = "10.18653/v1/2026.eacl-long.137",
    pages     = "2986--3004",
    ISBN      = "979-8-89176-380-7"
}
```

---

## Acknowledgments

This work was granted access to the HPC resources of **IDRIS** under the allocations 2023-AD011014882 and 2023-AD011014767, provided by **GENCI**.

This research was funded in whole or in part by **l'Agence Nationale de la Recherche (ANR)**, project ANR-22-CE38-0004.

---

## License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).