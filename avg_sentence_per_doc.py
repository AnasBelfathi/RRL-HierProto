import os
import json
import math
from collections import defaultdict


def compute_basic_stats(values):
    """
    Calcule la moyenne, la valeur min, max, et l'écart type d'une liste de valeurs.
    Retourne un dictionnaire avec ces stats.
    """
    if not values:
        return {
            'count': 0,
            'avg': 0.0,
            'min': 0.0,
            'max': 0.0,
            'std': 0.0
        }
    c = len(values)
    s = sum(values)
    avg = s / c
    min_val = min(values)
    max_val = max(values)
    var = sum((v - avg) ** 2 for v in values) / c
    std = math.sqrt(var)

    return {
        'count': c,
        'avg': avg,
        'min': min_val,
        'max': max_val,
        'std': std
    }


def process_train_file(train_path):
    """
    Lit un fichier train.json et calcule le nombre moyen de phrases par document et le nombre de labels distincts.
    """
    with open(train_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    sentence_counts = []
    label_set = set()

    for doc in documents:
        annotations = doc.get('annotations', [])
        if not annotations:
            continue
        annotation = annotations[0]
        results = annotation.get('result', [])
        num_sentences = len(results)  # Nombre de phrases dans le document
        sentence_counts.append(num_sentences)

        for item in results:
            labels = item['value'].get('labels', [])
            label_set.update(labels)

    return compute_basic_stats(sentence_counts), len(label_set)


def main():
    base_path = 'datasets'  # Adaptez si nécessaire

    task_stats = {}
    for task_dir in os.listdir(base_path):
        task_path = os.path.join(base_path, task_dir)
        if not os.path.isdir(task_path):
            continue

        train_file = os.path.join(task_path, 'train.json')
        if os.path.exists(train_file):
            stats, num_labels = process_train_file(train_file)
            task_stats[task_dir] = (stats, num_labels)

    # Affichage des résultats
    for task, (stats, num_labels) in task_stats.items():
        print(f"=== {task} ===")
        print(f"Nombre moyen de phrases par document: {stats['avg']:.2f}")
        # print(f"  Min: {stats['min']}, Max: {stats['max']}, Écart-type: {stats['std']:.2f}")
        print(f"Nombre de labels distincts: {num_labels}\n")


if __name__ == "__main__":
    main()