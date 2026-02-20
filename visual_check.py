import cv2
import random
from pathlib import Path


SAMPLES = 40
WINDOW_NAME = "YOLO Visual Audit"

CLASS_NAMES = {
    0: "UnPavedRoad",
    1: "UnMarkedBump",
    2: "Other"
}



def yolo_to_xyxy(xc, yc, bw, bh, img_w, img_h):
    x1 = int((xc - bw / 2) * img_w)
    y1 = int((yc - bh / 2) * img_h)
    x2 = int((xc + bw / 2) * img_w)
    y2 = int((yc + bh / 2) * img_h)
    return x1, y1, x2, y2


def visual_overlay(data_dir):
    img_exts = (".jpg", ".jpeg", ".png")
    images = [p for p in data_dir.iterdir() if p.suffix.lower() in img_exts]

    sample_imgs = random.sample(images, min(SAMPLES, len(images)))

    for img_path in sample_imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w, _ = img.shape
        label_path = img_path.with_suffix(".txt")

        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    cls = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:5])

                    x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, bw, bh, w, h)

                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    
                    label = CLASS_NAMES.get(cls, f"class_{cls}")
                    cv2.putText(
                        img,
                        label,
                        (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

        cv2.imshow(WINDOW_NAME, img)
        key = cv2.waitKey(0)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()



if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    DATASET_DIR = BASE_DIR / "Dataset3Class"

    visual_overlay(DATASET_DIR)