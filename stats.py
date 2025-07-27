import os
import json
from collections import Counter, defaultdict

DATASET_DIR = "datasets"


def analyze_dataset(path):
    stats = {
        "num_documents": 0,
        "num_annotations": 0,
        "label_distribution": Counter()
    }

    with open(path, "r") as f:
        data = json.load(f)
        stats["num_documents"] = len(data)

        for doc in data:
            for ann in doc.get("annotations", []):
                for res in ann.get("result", []):
                    labels = res["value"].get("labels", [])
                    stats["num_annotations"] += 1
                    stats["label_distribution"].update(labels)

    return stats


def main():
    results = defaultdict(dict)

    for subfolder in os.listdir(DATASET_DIR):
        subfolder_path = os.path.join(DATASET_DIR, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        for split in ["train.json", "dev.json", "test.json"]:
            split_path = os.path.join(subfolder_path, split)
            if os.path.isfile(split_path):
                stats = analyze_dataset(split_path)
                results[subfolder][split] = stats

    # Affichage lisible
    for dataset, splits in results.items():
        print(f"\n📁 Dataset: {dataset}")
        for split, stats in splits.items():
            print(f"  📄 {split}:")
            print(f"    - Documents: {stats['num_documents']}")
            print(f"    - Annotations: {stats['num_annotations']}")
            print(f"    - Labels:")
            for label, count in stats["label_distribution"].items():
                print(f"        {label}: {count}")


if __name__ == "__main__":
    main()
