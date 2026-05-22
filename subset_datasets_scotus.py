import os
import json

# Dossier racine contenant les sous-dossiers
root_dir = "datasets"

# Parcours des sous-dossiers commençant par 'scotus-'
for folder_name in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder_name)

    if os.path.isdir(folder_path) and folder_name.startswith("scotus-"):
        print(f"🔍 Traitement du dossier : {folder_name}")

        for split in ["train", "dev", "test"]:
            file_path = os.path.join(folder_path, f"{split}.json")

            if os.path.isfile(file_path):
                # Lecture du fichier
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Extraction des 5 premiers documents
                reduced_data = data[:5]

                # Écrasement du fichier avec les 5 documents
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(reduced_data, f, indent=2, ensure_ascii=False)

                print(f"✅ {file_path} réduit à 5 documents")
            else:
                print(f"⚠️ Fichier introuvable : {file_path}")
