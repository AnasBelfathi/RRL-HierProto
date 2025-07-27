import json

# === PARAMÈTRES ===
doc_id_target = "ad0e9fb9-da08-5c62-ad88-c649b39ad1b3"  # ← À adapter
path_reps = "reps_output.jsonl"
path_test_json = "datasets/legal-eval/test.json"
output_path = f"representations_with_labels_{doc_id_target}.json"

# === 1. Charger les représentations depuis reps_output.jsonl ===
reps_by_doc = {}

with open(path_reps, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        doc_id = obj["doc_id"]
        if doc_id not in reps_by_doc:
            reps_by_doc[doc_id] = []
        reps_by_doc[doc_id].append(obj["embedding"])

if doc_id_target not in reps_by_doc:
    raise ValueError(f"[!] Aucune représentation trouvée pour le doc_id '{doc_id_target}' dans {path_reps}")

# === 2. Récupérer le texte et les labels du document dans test.json ===
with open(path_test_json, "r", encoding="utf-8") as f:
    test_docs = json.load(f)

doc = next((d for d in test_docs if d["id"] == doc_id_target), None)
if not doc:
    raise ValueError(f"[!] doc_id '{doc_id_target}' non trouvé dans {path_test_json}")

# On récupère les annotations de phrases et labels
results = doc["annotations"][0]["result"]
phrases = []
for r in results:
    text = r["value"]["text"].strip()
    label = r["value"]["labels"][0] if r["value"]["labels"] else "UNKNOWN"
    phrases.append({"text": text, "label": label})

# === 3. Fusion représentation + label + texte ===
embeddings = reps_by_doc[doc_id_target]

if len(phrases) != len(embeddings):
    print(f"[⚠️] Attention : {len(phrases)} phrases vs {len(embeddings)} embeddings.")
    min_len = min(len(phrases), len(embeddings))
    phrases = phrases[:min_len]
    embeddings = embeddings[:min_len]

output = {
    "doc_id": doc_id_target,
    "sentences": [
        {
            "text": phrases[i]["text"],
            "label": phrases[i]["label"],
            "embedding": embeddings[i]
        }
        for i in range(len(phrases))
    ]
}

# === 4. Sauvegarde ===
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print(f"[✓] Sauvegarde terminée : {output_path}")
