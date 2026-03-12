"""
Train Bayesian portion estimator.
Fits Gaussian distribution parameters from annotated portion data.

Usage:
    python training/train_portion_estimator.py --csv ./datasets/portion_annotations.csv
"""
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Train portion estimator")
    parser.add_argument("--csv", type=str, required=True, help="Path to annotated portion CSV")
    parser.add_argument("--output", type=str, default="./data/portion_priors.json")
    return parser.parse_args()


def main():
    args = parse_args()
    import pandas as pd
    import numpy as np

    df = pd.read_csv(args.csv)
    # Expected columns: food_name, portion_grams
    priors = {}
    for food_name, group in df.groupby("food_name"):
        grams = group["portion_grams"].values
        priors[food_name] = {
            "mean_grams": float(np.mean(grams)),
            "std_grams": float(np.std(grams)),
            "min_grams": float(np.min(grams)),
            "max_grams": float(np.max(grams)),
        }
    with open(args.output, "w") as f:
        json.dump(priors, f, indent=2)
    print(f"Saved portion priors for {len(priors)} foods to {args.output}")


if __name__ == "__main__":
    main()
