"""
build_dataset.py — Collect a starter food image dataset.

Modes:
  1. scrape   — Download images using Google Image Search (icrawler)
  2. roboflow — Download the African Food Detection dataset from Roboflow
  3. both     — Do both (recommended)

Usage:
  # Scrape only
  python scripts/build_dataset.py --mode scrape --max_images 100

  # Roboflow only (needs API key)
  python scripts/build_dataset.py --mode roboflow --api_key YOUR_KEY

  # Both (recommended)
  python scripts/build_dataset.py --mode both --api_key YOUR_KEY --max_images 100
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path


# ── Priority foods to scrape (start manageable, covers everyday meals) ───────
PRIORITY_FOODS = [
    # Nigerian – directly in the DB
    "jollof rice",
    "egusi soup",
    "akara",
    "moi moi",
    "suya",
    "efo riro",
    "ewa agoyin",
    "nkwobi",
    "pepper soup",
    "okra soup",
    "fried plantain",
    "pounded yam",
    "amala",
    "eba",
    "fufu",
    "fried rice Nigerian",
    "chin chin Nigerian snack",
    "puff puff Nigerian",
    "boli Nigerian",
    "zobo drink",
    # International (common uploads)
    "rice and chicken",
    "pizza",
    "burger",
    "fried chicken",
    "salad",
    "pasta",
    "sushi",
    "sandwich",
    "omelette",
    "soup bowl",
]

# Map search term → folder/class name
def search_to_class(search_term: str) -> str:
    return (
        search_term.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def scrape_images(output_dir: Path, max_images: int = 100):
    """Download images via Bing Image Search (Google no longer works with icrawler)."""
    from icrawler.builtin import BingImageCrawler

    print(f"\n{'='*60}")
    print(f"SCRAPING {len(PRIORITY_FOODS)} food categories (via Bing)")
    print(f"Target: {max_images} images each")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    failed = []
    total_downloaded = 0

    for i, food in enumerate(PRIORITY_FOODS, 1):
        class_name = search_to_class(food)
        save_path = output_dir / class_name
        save_path.mkdir(parents=True, exist_ok=True)

        # Skip if already has enough images
        existing = list(save_path.glob("*.jpg")) + list(save_path.glob("*.png"))
        if len(existing) >= max_images * 0.8:
            print(f"[{i}/{len(PRIORITY_FOODS)}] SKIP {class_name} — {len(existing)} images already exist")
            total_downloaded += len(existing)
            continue

        print(f"[{i}/{len(PRIORITY_FOODS)}] Downloading: '{food}' → {class_name}/", end="", flush=True)
        before = len(existing)
        try:
            crawler = BingImageCrawler(
                storage={"root_dir": str(save_path)},
                log_level=50,  # suppress verbose logs
            )
            crawler.crawl(
                keyword=f"{food} food photo",
                max_num=max_images,
                file_idx_offset=before,
                filters={"type": "photo"},
            )
            after = len(list(save_path.glob("*.jpg")) + list(save_path.glob("*.png")))
            got = after - before
            total_downloaded += after
            print(f"  ✓ {got} new images ({after} total)")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed.append(food)

        time.sleep(1.5)  # polite delay between searches

    print(f"\nScraping complete. ~{total_downloaded} images across {len(PRIORITY_FOODS)} classes.")
    if failed:
        print(f"Failed categories ({len(failed)}): {failed}")


def download_roboflow(output_dir: Path, api_key: str):
    """Download African Food Detection dataset from Roboflow."""
    from roboflow import Roboflow

    print(f"\n{'='*60}")
    print("DOWNLOADING: African Food Detection (Roboflow)")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace("food-detection-93c3l").project("african-food-detection")
        dataset = project.version(1).download("yolov8", location=str(output_dir))
        print(f"\n✓ Roboflow dataset downloaded to: {output_dir}")
        print("  Contains: Akara, Efo-Riro, Egusi-Soup, Ewa-Agoyin, Jollof-Rice,")
        print("            Moi-Moi, Nkwobi, Okra-Soup, Pepper-Soup, Suya")
        return True
    except Exception as e:
        print(f"\n✗ Roboflow download failed: {e}")
        print("  → Get a free API key at: https://app.roboflow.com → Settings → API Keys")
        return False


def build_classifier_structure(scraped_dir: Path, output_dir: Path, val_split: float = 0.15):
    """
    Organise scraped images into train/val folder structure
    needed by train_classifier.py.

    datasets/food-clf/
      train/jollof_rice/001.jpg
      val/jollof_rice/002.jpg
    """
    import random

    print(f"\n{'='*60}")
    print("BUILDING CLASSIFIER DATASET STRUCTURE")
    print(f"Source: {scraped_dir}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    for class_dir in sorted(scraped_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        images = (
            list(class_dir.glob("*.jpg"))
            + list(class_dir.glob("*.jpeg"))
            + list(class_dir.glob("*.png"))
        )
        if not images:
            continue

        random.shuffle(images)
        n_val = max(1, int(len(images) * val_split))

        for i, img in enumerate(images):
            split = "val" if i < n_val else "train"
            dest = output_dir / split / class_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy(img, dest / img.name)

        print(f"  {class_dir.name:30s} → {len(images) - n_val} train, {n_val} val")

    print(f"\n✓ Classifier dataset ready at: {output_dir}")


def print_summary(base_dir: Path):
    """Print dataset statistics."""
    print(f"\n{'='*60}")
    print("DATASET SUMMARY")
    print(f"{'='*60}")

    for sub in sorted(base_dir.iterdir()):
        if sub.is_dir():
            imgs = (
                list(sub.rglob("*.jpg"))
                + list(sub.rglob("*.jpeg"))
                + list(sub.rglob("*.png"))
            )
            print(f"  {sub.name:30s}: {len(imgs):>5} images")

    print()


def main():
    parser = argparse.ArgumentParser(description="Build food image dataset")
    parser.add_argument(
        "--mode",
        choices=["scrape", "roboflow", "both"],
        default="scrape",
        help="Collection method",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="",
        help="Roboflow API key (required for roboflow/both modes)",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=100,
        help="Max images per class when scraping (default: 100)",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="./datasets",
        help="Root directory for all datasets",
    )
    args = parser.parse_args()

    base = Path(args.base_dir)
    scraped_dir = base / "scraped"
    roboflow_dir = base / "african-food"
    clf_dir = base / "food-clf"

    # ── Step 1: Collect images ─────────────────────────────────────────────
    if args.mode in ("scrape", "both"):
        scrape_images(scraped_dir, max_images=args.max_images)

    if args.mode in ("roboflow", "both"):
        if not args.api_key:
            print("\n⚠️  --api_key is required for roboflow mode.")
            print("   Get a free key at: https://app.roboflow.com → Settings → API Keys")
        else:
            download_roboflow(roboflow_dir, args.api_key)

    # ── Step 2: Build classifier folder structure from scraped images ──────
    if scraped_dir.exists():
        build_classifier_structure(scraped_dir, clf_dir)

    # ── Summary ───────────────────────────────────────────────────────────
    if base.exists():
        print_summary(base)

    print("Next steps:")
    print("  1. pip install -r requirements-training.txt")
    print("  2. python training/train_classifier.py --data_dir ./datasets/food-clf --epochs 30")
    if (roboflow_dir / "data.yaml").exists():
        print("  3. python training/train_detector.py --data ./datasets/african-food/data.yaml --epochs 50")


if __name__ == "__main__":
    main()
