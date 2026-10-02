import json

input_file = "synthetic_phishing_advanced_2026_clean.jsonl"   # change to your actual file name
output_file = "dataset_clean1.jsonl"                     # choose an output name

with open(input_file, "r", encoding="utf-8") as inf, \
     open(output_file, "w", encoding="utf-8") as outf:

    for line in inf:
        try:
            record = json.loads(line.strip())
            # Extract only the fields you need
            new_record = {
                "text": record["text"],
                "label": record["label"]
            }
            outf.write(json.dumps(new_record, ensure_ascii=False) + "\n")
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON line: {line[:100]}...")

print(f"Cleaned dataset written to {output_file}")