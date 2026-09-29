"""
Build the "legal-eval-v2" dataset split used throughout this repo.

Source: a Rhetorical-Role-annotated corpus of Indian court judgments in the
LegalEval@SemEval2023 label-studio JSON format ({"id", "annotations": [...],
"data": {"text": ...}} per document, 13 role labels: PREAMBLE, FAC, ISSUE,
ARG_RESPONDENT, ARG_PETITIONER, ANALYSIS, PRE_RELIED, PRE_NOT_RELIED, STA,
RLC, RPC, RATIO, NONE). You need your own copy of such a corpus with a
train.json and a dev.json (247 + 30 documents in the version this repo was
built against); point --src at the directory containing them.

Split logic:
  - the source dev.json is used AS-IS as the new test.json (never touched
    during training/model selection)
  - the source train.json is re-split into a new train/dev, at the same
    dev:train+dev ratio as this repo's original "legal-eval" dataset
    (30/277 ~= 10.8%), with a fixed seed for reproducibility.

Usage:
    python data_prep/build_legal_eval_v2.py --src /path/to/RR_data --dst ssc-datasets/legal-eval-v2
"""
import argparse
import json
import random
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Directory containing the source train.json and dev.json")
    ap.add_argument("--dst", default="ssc-datasets/legal-eval-v2", help="Output directory")
    ap.add_argument("--dev-fraction", type=float, default=30 / 277,
                    help="Fraction of the source train set held out as the new dev set "
                         "(default: matches this repo's original legal-eval train:dev ratio)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    train_full = json.load(open(src / "train.json"))
    n = len(train_full)
    n_dev = round(n * args.dev_fraction)

    indices = list(range(n))
    random.Random(args.seed).shuffle(indices)
    dev_idx = set(indices[:n_dev])

    new_train = [d for i, d in enumerate(train_full) if i not in dev_idx]
    new_dev = [d for i, d in enumerate(train_full) if i in dev_idx]
    assert len(new_train) + len(new_dev) == n

    with open(dst / "train.json", "w", encoding="utf-8") as f:
        json.dump(new_train, f, ensure_ascii=False)
    with open(dst / "dev.json", "w", encoding="utf-8") as f:
        json.dump(new_dev, f, ensure_ascii=False)

    shutil.copyfile(src / "dev.json", dst / "test.json")
    test_docs = json.load(open(dst / "test.json"))

    print(f"train: {len(new_train)} docs -> {dst}/train.json")
    print(f"dev:   {len(new_dev)} docs -> {dst}/dev.json")
    print(f"test:  {len(test_docs)} docs -> {dst}/test.json  (source dev.json, unchanged)")


if __name__ == "__main__":
    main()
