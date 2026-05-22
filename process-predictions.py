#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import csv
import os
from collections import defaultdict


def extract_text_from_test_json(test_data):
    """
    Construit une structure de données permettant de récupérer la i-ème phrase (ou segment)
    d'un document, étant donné un doc_id (ex. '2000_C_151') et un index (ex. 0).

    Retourne un dictionnaire:
    {
       doc_id: [ text_du_segment_0, text_du_segment_1, ... ]
    }
    """
    doc_to_sentences = defaultdict(list)

    for doc in test_data:
        doc_id = doc["id"]
        annotations = doc.get("annotations", [])
        if not annotations:
            continue

        results = annotations[0].get("result", [])
        for r in results:
            text_segment = r["value"]["text"]
            doc_to_sentences[doc_id].append(text_segment)

    return doc_to_sentences


def create_csv_for_task(task_name, strategies, base_inference_dir, base_dataset_dir, output_dir):
    """
    Génère un fichier CSV pour la tâche `task_name`, avec l'association correcte des segments de texte aux prédictions.

    - Colonne file_id = doc_name + index (ex. '2000_C_151/0')
    - Colonne text : texte récupéré via datasets
    - Colonne target : y_true (issu de la première stratégie trouvée)
    - Une colonne par stratégie : label prédit (y_predicted)
    """

    test_json_path = os.path.join(base_dataset_dir, task_name, "test.json")
    if not os.path.exists(test_json_path):
        print(f"[ATTENTION] test.json pour '{task_name}' introuvable: {test_json_path}")
        return

    with open(test_json_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    doc_to_sentences = extract_text_from_test_json(test_data)
    docs_info = {}

    found_first_strategy = False
    for strategy in strategies:
        pred_json_path = os.path.join(
            base_inference_dir,
            task_name,
            strategy,
            "seed_1",
            f"{task_name}_test_predictions.json"
        )

        if not os.path.exists(pred_json_path):
            print(f"[INFO] Fichier de prédictions introuvable pour '{task_name}/{strategy}': {pred_json_path}")
            continue

        with open(pred_json_path, "r", encoding="utf-8") as pf:
            predictions = json.load(pf)

        doc_names = predictions.get("doc_names", [])
        docwise_y_true = predictions.get("docwise_y_true", [])
        docwise_y_pred = predictions.get("docwise_y_predicted", [])

        if not found_first_strategy:
            found_first_strategy = True

        for doc_idx, doc_name in enumerate(doc_names):
            if doc_name not in doc_to_sentences:
                print(f"[WARNING] doc_name '{doc_name}' n'a pas de correspondance dans test.json")
                continue

            segments = doc_to_sentences[doc_name]

            if doc_idx >= len(docwise_y_true) or doc_idx >= len(docwise_y_pred):
                print(f"[WARNING] Taille des labels incorrecte pour {doc_name}")
                continue

            y_true_segments = docwise_y_true[doc_idx]
            y_pred_segments = docwise_y_pred[doc_idx]

            if len(segments) != len(y_true_segments):
                print(f"[WARNING] Nombre de segments différent entre test.json ({len(segments)}) et prédictions ({len(y_true_segments)}) pour {doc_name}")
                continue

            for i, (text, y_true, y_pred) in enumerate(zip(segments, y_true_segments, y_pred_segments)):
                segment_id = f"{doc_name}/{i}"

                if segment_id not in docs_info:
                    docs_info[segment_id] = {
                        "file_id": segment_id,
                        "text": text,
                        "target": y_true,
                    }

                docs_info[segment_id][strategy] = y_pred

    columns = ["file_id", "text", "target"] + strategies
    os.makedirs(output_dir, exist_ok=True)

    csv_filename = os.path.join(output_dir, f"{task_name}_predictions.csv")
    with open(csv_filename, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        for info in docs_info.values():
            writer.writerow(info)

    print(f"[OK] CSV généré pour la tâche '{task_name}' : {csv_filename}")


if __name__ == "__main__":
    tasks = ["scotus-rhetorical_function"]
    strategies = ["baseline-WS_3-CRF_False-SentenceLSTM_False", "baseline-WS_3-CRF_False-SentenceLSTM_True", "self_attention_context-WS_4-CRF_False-SentenceLSTM_True"]

    base_inference_dir = "output-training"
    base_dataset_dir = "datasets"
    output_dir = "analysis-predictions"

    for task in tasks:
        create_csv_for_task(
            task_name=task,
            strategies=strategies,
            base_inference_dir=base_inference_dir,
            base_dataset_dir=base_dataset_dir,
            output_dir=output_dir
        )
