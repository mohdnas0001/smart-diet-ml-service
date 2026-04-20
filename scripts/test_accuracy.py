"""Test per-class prediction accuracy against the scraped dataset."""
import subprocess, json, sys
from pathlib import Path
from collections import Counter

# Same aliases as food_mapper.py — folder name → canonical DB name
FOLDER_TO_CANONICAL = {
    "boli_nigerian":            "boli",
    "chin_chin_nigerian_snack": "chin_chin",
    "puff_puff_nigerian":       "puff_puff",
    "moi_moi":                  "moin_moin",
    "fried_rice_nigerian":      "nigerian_fried_rice",
    "zobo_drink":               "zobo",
    "salad":                    "nigerian_salad",
    "rice_and_chicken":         "fried_rice",
    "pasta":                    "jollof_spaghetti",
    "pizza":                    "nigerian_shawarma",
    "sandwich":                 "meat_pie",
    "sushi":                    "fried_fish",
    "soup_bowl":                "pepper_soup",
    "burger":                   "meat_pie",
    "omelette":                 "scotch_egg",
}

scraped = Path("datasets/scraped")
results = []

for food_dir in sorted(scraped.iterdir()):
    if not food_dir.is_dir():
        continue
    raw_name = food_dir.name
    # What should the API return for this folder?
    expected = FOLDER_TO_CANONICAL.get(raw_name, raw_name)
    food = raw_name  # keep original for display
    images = list(food_dir.glob("*.jpg"))[:10]
    correct = 0
    wrong_preds = []

    for img in images:
        try:
            r = subprocess.run(
                ["curl", "-s", "-X", "POST", "http://localhost:8000/api/predict",
                 "-F", f"file=@{img}"],
                capture_output=True, text=True, timeout=15
            )
            data = json.loads(r.stdout)
            pred = data["food_items"][0]["name"]
            if pred == expected:
                correct += 1
            else:
                wrong_preds.append(pred)
        except Exception as e:
            pass

    total = len(images)
    if total == 0:
        continue
    acc = correct * 100 // total
    icon = "✅" if acc >= 80 else ("⚠️ " if acc >= 50 else "❌")
    expected_label = f"({expected})" if expected != food else ""
    wrong_str = ""
    if wrong_preds:
        top = Counter(wrong_preds).most_common(2)
        wrong_str = "  → got: " + ", ".join(f"{n}({c}x)" for n,c in top)
    print(f"{icon} {food:<35} {correct}/{total}  ({acc}%) {expected_label}{wrong_str}")
    results.append((acc, food))

if results:
    avg = sum(a for a, _ in results) // len(results)
    good = sum(1 for a, _ in results if a >= 80)
    bad  = sum(1 for a, _ in results if a < 50)
    print(f"\n{'='*70}")
    print(f"Overall avg: {avg}%  |  Good (≥80%): {good}  |  Needs work (<50%): {bad}  |  Classes: {len(results)}")
    print(f"\nWeakest classes:")
    for acc, food in sorted(results)[:5]:
        print(f"  {food}: {acc}%")
