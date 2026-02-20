import os
import cv2
import glob
from pathlib import Path


def audit_dataset_in_place(data_dir):
    """
    Clean and verify a YOLO-style dataset directory.
    - Verifies image readability
    - Checks label presence
    - Clips YOLO coords to [0, 1]
    - Rewrites cleaned labels in place
    """

    # 1. Initialize image paths list (FIXED)
    image_extensions = ("*.jpg", "*.jpeg", "*.png")
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(data_dir, ext)))

    stats = {
        "total": len(image_paths),
        "corrupt": 0,
        "missing_lbl": 0,
        "fixed": 0
    }

    print(f"Auditing {stats['total']} files in {data_dir}...")

    for img_path in image_paths:
        # Check if image is readable
        img = cv2.imread(img_path)
        if img is None:
            stats["corrupt"] += 1
            continue

        # Match label file in same folder
        basename = Path(img_path).stem
        lbl_path = os.path.join(data_dir, f"{basename}.txt")

        if not os.path.exists(lbl_path):
            stats["missing_lbl"] += 1
            continue

        with open(lbl_path, "r") as f:
            lines = f.readlines()

        # 2. Initialize valid_lines (FIXED)
        valid_lines = []

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            # 3. Correct class ID extraction (FIXED)
            cls_id = parts[0]

            try:
                # YOLO: x_center, y_center, width, height
                coords = [float(x) for x in parts[1:5]]
            except ValueError:
                # Skip malformed numeric entries
                stats["fixed"] += 1
                continue

            # Clip values to legal YOLO bounds [0, 1]
            clipped = False
            for i, c in enumerate(coords):
                if c < 0.0 or c > 1.0:
                    coords[i] = max(0.0, min(1.0, c))
                    clipped = True

            if clipped:
                stats["fixed"] += 1

            new_line = f"{cls_id} " + " ".join(f"{c:.6f}" for c in coords) + "\n"
            valid_lines.append(new_line)

        # Overwrite with clean data
        with open(lbl_path, "w") as f:
            f.writelines(valid_lines)

    print(f"\nAudit Summary for {data_dir}:")
    for key, val in stats.items():
        print(f" - {key}: {val}")


# To run:
audit_dataset_in_place("/Users/khushichauhan/armsproject/Dataset3class")