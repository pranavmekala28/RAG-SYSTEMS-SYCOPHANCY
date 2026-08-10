from datasets import load_dataset
import json, os

SUBSET_DOMAIN_MAP = {
    "finqa":    "finance",
    "tatqa":    "finance",
    "pubmedqa": "medical",
    "covidqa":  "medical",
    "expertqa": "science",
    "techqa":   "technology",
    "emanual":  "technology",
}

os.makedirs("data", exist_ok=True)

sample_row = None
file_stats = []

for subset, domain in SUBSET_DOMAIN_MAP.items():
    out_path = f"data/{domain}_{subset}.jsonl"
    try:
        ds = load_dataset("galileo-ai/ragbench", subset, split="test[:200]")
        with open(out_path, "w", encoding="utf-8") as f:
            for row in ds:
                f.write(json.dumps(row) + "\n")
                if sample_row is None:
                    sample_row = row
        file_stats.append((out_path, len(ds)))
    except Exception as e:
        print(f"ERROR loading {subset}: {e}")
        continue

print("\nFile row counts:")
for path, count in file_stats:
    print(f"  {path}: {count}")

if sample_row:
    print("\nColumn names from one row:")
    print(list(sample_row.keys()))
