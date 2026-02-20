import random
import shutil
from pathlib import Path

N = 100
SEED = 42

random.seed(SEED)

SRC = Path("Dataset3Class")
DST = Path("calib_images")

DST.mkdir(exist_ok=True)

images = [p for p in SRC.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
subset = random.sample(images, min(N, len(images)))

for img in subset:
    shutil.copy2(img, DST / img.name)

print(f"Calibration set created with {len(list(DST.iterdir()))} images")