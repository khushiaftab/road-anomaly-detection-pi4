import random
import shutil
from pathlib import Path


SPLIT_RATIO = 0.8   
SEED = 42          



def split_dataset():
    random.seed(SEED)

    base_dir = Path(__file__).resolve().parent
    src_dir = base_dir / "Dataset3Class"

    out_dir = base_dir / "dataset"
    img_train = out_dir / "images" / "train"
    img_val   = out_dir / "images" / "val"
    lbl_train = out_dir / "labels" / "train"
    lbl_val   = out_dir / "labels" / "val"

    for d in [img_train, img_val, lbl_train, lbl_val]:
        d.mkdir(parents=True, exist_ok=True)

    
    img_exts = {".jpg", ".jpeg", ".png"}
    images = [p for p in src_dir.iterdir() if p.suffix.lower() in img_exts]

    images.sort()
    random.shuffle(images)

    split_idx = int(len(images) * SPLIT_RATIO)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    def copy_pair(img_path, img_dst, lbl_dst):
        lbl_path = img_path.with_suffix(".txt")
        if not lbl_path.exists():
            return
        shutil.copy2(img_path, img_dst / img_path.name)
        shutil.copy2(lbl_path, lbl_dst / lbl_path.name)

    for img in train_imgs:
        copy_pair(img, img_train, lbl_train)

    for img in val_imgs:
        copy_pair(img, img_val, lbl_val)

    print("Split complete.")
    print(f"Train images: {len(train_imgs)}")
    print(f"Val images:   {len(val_imgs)}")


if __name__ == "__main__":
    split_dataset()