import os
import json
import pandas as pd

# 📁 Racine du dossier contenant les données
ROOT_DIR = "./matching-context/new_similarity_outputs_with_labels/none/mean"

# 📂 Liste des sous-dossiers à traiter (tâches SCOTUS)
scotus_tasks = ["scotus-category", "scotus-rhetorical_function", "scotus-steps"]

# 📁 Dossier de sortie
output_dir = "./proto_predictions_extraction_csv_format"
os.makedirs(output_dir, exist_ok=True)

for task in scotus_tasks:
    task_dir = os.path.join(ROOT_DIR, task)
    all_rows = []

    # 🔄 On traite tous les fichiers jsonl du dossier (train/dev/test)
    for fname in ["test.jsonl"]:
        file_path = os.path.join(task_dir, fname)
        if not os.path.exists(file_path):
            continue

        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line)
                row = {
                    "doc_id": data["doc_id"],
                    "sentence_idx": data["sentence_idx"],
                    "sentence_text": data["sentence_text"],
                    "true_label": data["true_label"],
                    "closest_centroid_label": data["closest_centroid_label"]
                }
                all_rows.append(row)

    # ✅ Créer un DataFrame et sauvegarder
    df = pd.DataFrame(all_rows)
    output_csv = os.path.join(output_dir, f"{task}.csv")
    df.to_csv(output_csv, index=False)
    print(f"{task} ✅ sauvegardé : {output_csv} ({len(df)} lignes)")
