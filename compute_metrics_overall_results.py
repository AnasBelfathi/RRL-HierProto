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
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    return macro_f1, weighted_f1, micro_f1

def process_predictions(base_dir, output_file):
    """
    Parcourt les fichiers de prédictions dans la structure donnée et calcule les métriques.
    """
    results = []
    missing_files = []

    print(base_dir)
    for task in os.listdir(base_dir):
        task_path = os.path.join(base_dir, task)
        print(task_path)
        if not os.path.isdir(task_path):
            continue

        for context in os.listdir(task_path):
            context_path = os.path.join(task_path, context)
            if not os.path.isdir(context_path):
                continue

            # for method in os.listdir(context_path):
            #     method_path = os.path.join(context_path, method)
            #     print(method_path)
            #     if not os.path.isdir(method_path):
            #         continue

            for seed in os.listdir(context_path):
                if not seed.startswith("seed_"):
                    continue  # Ne traiter que les dossiers seed_x
                seed_path = os.path.join(context_path, seed)
                if not os.path.isdir(seed_path):
                    continue

                # Parcourir tous les fichiers de prédictions
                for file in os.listdir(seed_path):
                    if file.endswith("_test_predictions.json"):
                        file_path = os.path.join(seed_path, file)

                        with open(file_path, 'r') as f:
                            data = json.load(f)

                        y_true = data.get("y_true", [])
                        y_pred = data.get("y_predicted", [])

                        if not y_true or not y_pred:
                            print(f"Missing labels in {file_path}")
                            continue

                        # Calcul des métriques
                        macro_f1, weighted_f1, micro_f1 = calculate_metrics(y_true, y_pred)

                        results.append({
                            "task": task,
                            "context": context,
                            # "method": method,
                            "seed": seed,
                            "macro_f1": macro_f1,
                            "weighted_f1": weighted_f1,
                            "micro_f1" : micro_f1
                        })

    # Sauvegarde des résultats sous forme de DataFrame
    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results_df.to_csv(output_file, index=False)

    # Afficher les fichiers manquants
    if missing_files:
        print("Fichiers manquants :")
        for missing in missing_files:
            print(f"Tâche: {missing[0]}, Contexte: {missing[1]}, Méthode: {missing[2]}, Seed: {missing[3]}")

    print(f"Résultats sauvegardés dans {output_file}")

if __name__ == "__main__":
    BASE_DIR = "outputs/injection_strategies_output"  # Répertoire d'output
    OUTPUT_FILE = "csv_scores/injection_strategies_output.csv"  # Fichier CSV pour les résultats

    process_predictions(BASE_DIR, OUTPUT_FILE)

























