"""Rebuild datasets/food-clf train/val split from datasets/scraped/."""
import shutil, random
from pathlib import Path

scraped   = Path("datasets/scraped")
output    = Path("datasets/food-clf")
VAL_SPLIT = 0.15

# Wipe old split
if output.exists():
    shutil.rmtree(output)
    print("Cleared old food-clf")

total_train = total_val = 0
for cls_dir in sorted(scraped.iterdir()):
    if not cls_dir.is_dir():
        continue
    imgs = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
    if not imgs:
        continue
    random.shuffle(imgs)
    n_val = max(1, int(len(imgs) * VAL_SPLIT))
    for i, img in enumerate(imgs):
        split = "val" if i < n_val else "train"
        dest = output / split / cls_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(img, dest / img.name)
    n_train = len(imgs) - n_val
    total_train += n_train
    total_val   += n_val
    print(f"  {cls_dir.name:<35} train={n_train}  val={n_val}")

print(f"\nDone. Total: {total_train} train, {total_val} val across {len(list(output.iterdir()))} splits")
