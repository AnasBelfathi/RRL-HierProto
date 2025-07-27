# 🏛️ Coupling Local Context and Global Semantic Prototypes via Hierarchical Architecture for Rhetorical Roles Labeling

This repository supports the experiments from the paper:

**Coupling Local Context and Global Semantic Prototypes via Hierarchical Architecture for Rhetorical Roles Labeling**  
*(Currently under review process)*


---

## 📚 SCOTUS-LAW Corpus

We introduce **SCOTUS-LAW**, a new benchmark corpus for **rhetorical roles labeling** in U.S. Supreme Court decisions.  
The dataset is available in the `datasets/` directory and is organized according to three levels of annotation:

- `scotus-category/`: high-level argumentative categories  
- `scotus-rhetorical_function/`: rhetorical functions per sentence  
- `scotus-steps/`: hierarchical annotation that contains additional attributes  

Each subfolder contains three files: `train.json`, `dev.json`, and `test.json`, with annotations at the sentence level.

---

## 🧪 Running Experiments

To reproduce our baseline experiments, run:

```bash
python baseline_run.py
