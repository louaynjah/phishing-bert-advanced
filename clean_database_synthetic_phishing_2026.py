#!/usr/bin/env python3
"""
Extrait uniquement text et label du CSV généré par data_aug1.py
et les sauvegarde au format JSONL.
"""

import csv
import json

# ============================================================
# CONFIGURATION
# ============================================================
INPUT_CSV = "synthetic_phishing_2026.csv"   # À adapter
OUTPUT_JSONL = "dataset_clean.jsonl"        # Nom de sortie

# ============================================================
# LECTURE / ÉCRITURE
# ============================================================
with open(INPUT_CSV, "r", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as out:
        for row in reader:
            # On prend le texte tel quel et le label
            new_record = {
                "text": row["text"],
                "label": int(row["label"])   # 0 ou 1
            }
            out.write(json.dumps(new_record, ensure_ascii=False) + "\n")

print(f"✅ Fichier généré : {OUTPUT_JSONL}")