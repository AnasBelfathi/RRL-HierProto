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

## About this repository

Reproduction code for prototype-based knowledge injection methods on
**Rhetorical Role Labeling (RRL)** — classifying each sentence of a legal
judgment by its rhetorical function (Preamble, Facts, Issue, Analysis,
Ratio, ...) — built on top of a **Hierarchical Sequential Labeling Network**
(HSLN: BERT → word BiLSTM → attention pooling → sentence BiLSTM → CRF).

Four configurations are compared, all trainable/evaluatable with the same
scripts:

| Method | What it does | Script |
|---|---|---|
| **Baseline** | Plain HSLN, no external knowledge injection | `baseline_run.py` |
| **PCM** (Prototype-Conditioned Modulation) | Injects a corpus-level prototype vector (mean embedding per rhetorical role, computed with an external encoder) into the sentence representation via a learned fusion module (concat/gated/FiLM/cross-attention/conditional-LayerNorm), before or after the sentence BiLSTM | `context_fusion.py`, `context-extraction/`, `matching-context/`, `baseline_run.py --ctx_fusion ... --ctx_position ...` |
| **PBR** (Prototype-Based Regularization) | Learns its own internal prototype bank (gradient-trained) and regularizes the sentence representation towards it via 2 auxiliary losses (clustering, separation), replacing the CRF head with a distance-based classifier | `prototype_net/`, `baseline_run.py --use_prototypes True` |
| **Mind-Your-Neighbours** (single prototype) | *Post-hoc*, no retraining: interpolates the trained baseline's own softmax with a kNN-style distribution over one prototype per label, λ/τ grid-searched on dev | `mind_neighbours_run.py` |

**Dataset-agnostic**: every script takes the task as a `--task`/`--dataset`
argument (or `$PCM_TASK` for the SLURM scripts) and reads its label list
from `task.py`. Adding a new dataset means adding one `task_type` entry
there plus raw JSON under `ssc-datasets/<task>/` — see
[Adding a new dataset](#adding-a-new-dataset). Two tasks are already
registered and were used to validate this pipeline: a legal-domain RRL
dataset (Indian court judgments, 13 rhetorical roles, referred to as
`legal-eval-v2` here; see [Data](#data)) and `scotus-rhetorical_function`
(US Supreme Court opinions, rhetorical-function labels) — the examples
below use `legal-eval-v2`/`scotus-rhetorical_function` purely as
placeholders for whichever task you plug in.

---

## Repository layout

```
models.py                  HSLN model (BertHSLN) + PCM fusion + PBR integration
context_fusion.py          PCM fusion modules (ConcatProjection, GatedAdd, FiLM, CrossAttention, ConditionalLayerNorm)
prototype_net/             PBR: learnable prototype layer + the 2 auxiliary losses
task.py                    Task/label definitions, fold/batch loading
dataset_reader.py, batch_creator.py, bucketing.py
                            Tokenized-data reading and batching for the HSLN model
train.py, eval.py, eval_run.py
                            Training loop, metrics (weighted/macro-F1), results.csv writer
baseline_run.py            CLI entry point: trains baseline / PCM / PBR
mind_neighbours_run.py     Standalone: trains a baseline, then single-prototype interpolation
process_datasets.py        Raw JSON -> HSLN-tokenized text format
data_prep/build_legal_eval_v2.py
                            Example data-prep script (builds the legal-eval-v2 split);
                            write your own here for a dataset that needs custom splitting
context-extraction/        Builds PCM's corpus-level prototypes (build_centroids.py)
matching-context/          Assigns each sentence its nearest PCM prototype (match_centroids.py)
document-grouping/         OpenAI-embedding + KMeans document clustering, for PCM's
                            "supervised-clustering" variant (openai_kmeans_clustering.py)
slurms/jeanzay/pipeline/   SLURM orchestration (written for Jean Zay/IDRIS, adapt the
                            module/account bits for another cluster) -- see below.
                            One job file per experiment: 00_build_centroids,
                            10_match_centroids, 20_baseline, 21_pcm, 22_pbr,
                            25_mind_proto
```


## Data

This repo does not redistribute any corpus. You need:

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

The three granularities map to the `scotus-category`, `scotus-rhetorical_function`,
and `scotus-steps` task identifiers registered in `task.py`.

> 📧 **Data Access:** The SCOTUS-LAW dataset is not included in this repository for privacy reasons. Please contact the authors at `anas.belfathi@univ-nantes.fr` and `nicolas.hernandez@univ-nantes.fr` to request access.


### Models

`bert-base-uncased` (HSLN backbone) and `nlpaueb/legal-bert-base-uncased`
(PCM's/document-grouping's embedding model), both from the Hugging Face Hub.

Either way, place the raw JSON under `ssc-datasets/<task>/{train,dev,test}.json`.

### Adding a new dataset

1. Format your annotations as label-studio-style JSON, one file per split:
   `ssc-datasets/<your-task>/{train,dev,test}.json`, each document
   `{"id": ..., "data": {"text": "..."}, "annotations": [{"result": [{"value": {"text": "<sentence>", "labels": ["<ROLE>"]}}, ...]}]}`.
2. Register it in `task.py`'s `pubmed_task()`: add an
   `elif task_type == "<your-task>":` branch with its label list (see the
   existing `legal-eval-v2`/`scotus-*` branches for the pattern).
3. (Optional, for PCM's `supervised-clustering` variant) add `<your-task>`
   to the `LEGAL_DATASETS` set in `context-extraction/build_centroids.py`
   and `matching-context/match_centroids.py` if it should use LegalBERT
   instead of SciBERT for embeddings.
4. Run the [Pipeline](#pipeline) below with `--task <your-task>` /
   `PCM_TASK=<your-task>` everywhere. If your raw JSON needs its own
   train/dev/test split logic (ours didn't come pre-split), write a small
   script under `data_prep/` the way `build_legal_eval_v2.py` does for the
   `legal-eval-v2` example — otherwise skip straight to step 1 of the
   pipeline with your already-split JSON.

## Pipeline

```
1. Tokenize            python process_datasets.py ssc-datasets/ processed-datasets/
2. PCM prototypes       python context-extraction/build_centroids.py --datasets <task> --emb_type none --strategy mean --use_labels
3. PCM matching         python matching-context/match_centroids.py --dataset <task> --centroids_dir context-extraction/proto-with-labels-v2/no_cluster/mean/<task> --emb_type none --strategy mean --out_root matching-context/new_similarity_outputs_with_labels
4. Train (any config)   python baseline_run.py --task <task> --seed <N> --tokenized_folder processed-datasets --output_dir <out> ...
   or                   python mind_neighbours_run.py --task <task> --seed <N> --tokenized_folder processed-datasets --output_dir <out>
```

`baseline_run.py`'s relevant flags:

```bash
# baseline (no injection)
--emb_type decoder --centroid_strategy mean --ctx_fusion concat_proj --ctx_position "" --use_crf True

# PCM (corpus-level prototypes, injected pre sentence-BiLSTM)
--emb_type none --centroid_strategy mean --ctx_fusion concat_proj --ctx_position pre

# PBR (learnable prototypes)
--use_prototypes True --proto_training joint --n_prototypes 8 --lambda_c 0.9 --lambda_s 0.9 --proto_dist euclidean
```

`--emb_type none|random-clusters|supervised-clustering` selects PCM's
document-grouping strategy (`none` = one prototype per label over the whole
corpus, no document clustering; the other two need
`document-grouping/openai_kmeans_clustering.py` — or your own
document-level clustering, same output schema — run first).

## SLURM orchestration (Jean Zay)

`slurms/jeanzay/pipeline/` has a ready-made pipeline for a SLURM
cluster (written for Jean Zay/IDRIS — adjust `env.sh`'s `module load` lines
and `--account` for another cluster):

All commands below run from `slurms/jeanzay/pipeline/` (`cd
$PCM_HOME/slurms/jeanzay/pipeline` once `PCM_HOME` is set, or wherever you
cloned this repo).

1. Edit `env.sh`: set `PCM_ACCOUNT`/`PCM_ACCOUNT_EXTRACT` to your allocation,
   `PCM_HOME` to where you cloned this repo.
2. `python download_models.py`: run once on a **frontend/login node**
   (compute nodes usually have no internet) to fetch `bert-base-uncased` and
   `nlpaueb/legal-bert-base-uncased` locally under `$PCM_HOME/models/`.
3. `bash smoke_test.sh` inside a small interactive allocation: runs the
   whole pipeline once at reduced scale (`--mini_data True`, 2 docs/split, 1
   epoch) for every method, on whichever task `$PCM_TASK` is set to, to
   catch environment/config issues before spending real compute.
4. `bash launch.sh <task> <seed> [seed...]` — submits baseline + PCM (3
   strategies) + PBR + Mind-proto for the given task/seeds (any
   task registered in `task.py`, see
   [Adding a new dataset](#adding-a-new-dataset)), with the right SLURM
   array sizes computed automatically. Re-running it for seeds you already
   have is safe/cheap: every step skips work whose `results.csv` already
   exists.


Numbered scripts: `00_build_centroids` → `10_match_centroids` → `21_pcm`
(chained by dependency); `20_baseline`, `22_pbr`, `25_mind_proto` are
independent of each other and of 00/10. `launch.sh` submits all of them
with the right dependencies for you.

## Troubleshooting

- **Compute nodes with no internet**: pre-download HF models on a frontend
  node (`download_models.py`), and `env.sh` sets `HF_HUB_OFFLINE=1` /
  `TRANSFORMERS_OFFLINE=1` so a missing model fails fast instead of hanging.
- **`~/.local` (pip `--user`) packages conflicting with your conda/venv
  env**: `env.sh` sets `PYTHONNOUSERSITE=1`. Symptom if this isn't set:
  `ValueError: numpy.dtype size changed, may indicate binary incompatibility`
  when importing allennlp/spacy/thinc.
- **`$0` inside a `sbatch` job**: on some clusters, `sbatch` runs a spooled
  copy of the script, so `dirname "$0"` doesn't resolve to the repo — the
  `.slurm` scripts fall back to `$SLURM_SUBMIT_DIR` for this reason.

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

