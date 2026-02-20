from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent / "Dataset3Class"

class_ids = set()

for txt_file in DATASET_DIR.glob("*.txt"):
    with open(txt_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 1:
                try:
                    class_ids.add(int(parts[0]))
                except ValueError:
                    pass

print("Class IDs found:", sorted(class_ids))
print("Total number of classes:", len(class_ids))