import json
import os

INPUT_FILE = "emails_expanded_2026_Gemini.json"      # change to your file name
OUTPUT_FILE = "dataset_clean2.jsonl"  # output

def combine_subject_body(record):
    subject = record.get("Subject", "")
    body = record.get("Body", "")
    # Combine with clear separation, preserving the original subject line
    return f"Subject: {subject}\n\n{body}"

def determine_label(record):
    # Type may be "Phishing" or something else; treat as phishing if Type contains "Phishing"
    type_val = record.get("Type", "")
    return 1 if "Phishing" in type_val else 0

def extract_metadata(record):
    # Keep all fields except Subject and Body (since they go into text)
    # You may also want to keep Subject separately if needed, but we already combine it.
    exclude = {"Subject", "Body", "No."}  # No. is not needed
    meta = {}
    for key, value in record.items():
        if key not in exclude:
            meta[key] = value
    return meta

# Read the dataset (assuming it's a JSON array)
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)  # data is a list of dicts

# Process and write to JSONL
with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for record in data:
        new_record = {
            "text": combine_subject_body(record),
            "label": determine_label(record),
            "metadata": extract_metadata(record)
        }
        out.write(json.dumps(new_record, ensure_ascii=False) + "\n")

print(f"Transformed {len(data)} records into {OUTPUT_FILE}")