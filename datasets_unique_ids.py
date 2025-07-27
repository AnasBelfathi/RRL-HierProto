import json
import os
import uuid

# Dossier racine contenant tous les sous-dossiers de datasets
root_dir = "datasets"

# On définit un namespace fixe (important pour la reproductibilité)
NAMESPACE_UUID = uuid.UUID('12345678-1234-5678-1234-567812345678')


# Fonction pour générer un ID unique basé sur un texte
def generate_deterministic_id(text):
    return str(uuid.uuid5(NAMESPACE_UUID, text))


# Parcourir tous les sous-dossiers
for dataset_name in os.listdir(root_dir):
    dataset_path = os.path.join(root_dir, dataset_name)

    if os.path.isdir(dataset_path):
        print(f"Traitement du dataset : {dataset_name}")

        for split in ["train.json", "dev.json", "test.json"]:
            input_path = os.path.join(dataset_path, split)

            if os.path.exists(input_path):
                output_path = os.path.join(dataset_path, f"{split}")

                with open(input_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                updated_data = []
                for doc in data:
                    # Nouveau doc ID basé sur le texte complet
                    new_doc_id = generate_deterministic_id(doc['data'])
                    doc['id'] = new_doc_id

                    for annotation in doc.get('annotations', []):
                        for result in annotation.get('result', []):
                            # Nouvel ID basé sur le texte de la phrase annotée
                            sentence_text = result['value']['text']
                            result['id'] = generate_deterministic_id(sentence_text)

                    updated_data.append(doc)

                # Sauvegarder
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(updated_data, f, indent=2, ensure_ascii=False)

                print(f"  ➔ Fichier {split} mis à jour -> {output_path}")

print("\n✅ Tous les datasets ont été traités avec reproductibilité.")
