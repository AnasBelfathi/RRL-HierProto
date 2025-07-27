import os
import json
from sklearn.metrics import f1_score
import pandas as pd

def calculate_metrics(y_true, y_pred):
    """
    Calcule les métriques Macro-F1 et Weighted-F1 à partir des étiquettes vraies et prédites.
    """
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    return macro_f1, weighted_f1

def process_predictions(base_dir, output_file):
    """
    Parcourt les fichiers de prédictions dans la structure donnée et calcule les métriques.
    """
    results = []
    missing_files = []

    for task in os.listdir(base_dir):
        task_path = os.path.join(base_dir, task)
        if not os.path.isdir(task_path):
            continue

        for context in os.listdir(task_path):
            context_path = os.path.join(task_path, context)
            if not os.path.isdir(context_path):
                continue

            for seed in os.listdir(context_path):
                if not seed.startswith("seed_"):
                    continue  # Ne traiter que les dossiers seed_x
                seed_path = os.path.join(context_path, seed)
                if not os.path.isdir(seed_path):
                    continue

                # Parcourir tous les fichiers de prédictions
                for file in os.listdir(seed_path):
                    if file.endswith("_predictions.json"):
                        file_path = os.path.join(seed_path, file)

                        with open(file_path, 'r') as f:
                            data = json.load(f)

                        y_true = data.get("y_true", [])
                        y_pred = data.get("y_predicted", [])

                        if not y_true or not y_pred:
                            print(f"Missing labels in {file_path}")
                            continue


                        # Calcul des métriques
                        macro_f1, weighted_f1 = calculate_metrics(y_true, y_pred)

                        results.append({
                            "task": task,
                            "context": context,
                            "seed": seed,
                            "macro_f1": macro_f1,
                            "weighted_f1": weighted_f1
                        })

    # Sauvegarde des résultats sous forme de DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)

    # Afficher les fichiers manquants
    if missing_files:
        print("Fichiers manquants :")
        for missing in missing_files:
            print(f"Tâche: {missing[0]}, Contexte: {missing[1]}, Seed: {missing[2]}")

    print(f"Résultats sauvegardés dans {output_file}")

if __name__ == "__main__":
    BASE_DIR = "output-training"  # Répertoire d'output
    OUTPUT_FILE = "scores/NEW_SCORES.csv"  # Fichier CSV pour les résultats

    process_predictions(BASE_DIR, OUTPUT_FILE)
