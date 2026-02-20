from pathlib import Path
from collections import Counter

DATASET_DIR = Path("Dataset3Class")

counter = Counter()

for txt in DATASET_DIR.glob("*.txt"):
    with open(txt) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                counter[int(parts[0])] += 1

print("Class distribution:")
total = sum(counter.values())
for cls, cnt in sorted(counter.items()):
    print(f"Class {cls}: {cnt} ({cnt/total:.1%})")