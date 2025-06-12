"""Prepare ECHI data"""

import json
import logging
from glob import glob

import hydra
import numpy as np
from omegaconf import DictConfig


def read_jsonl(file_path, data=None):
    """Read a JSONL file and return a list of dictionaries."""
    if data is None:
        data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


NON_NUMERIC_KEYS = {"key"}


def compute_stats(results):
    """Compute statistics from the results."""
    stats = dict()
    valid_keys = [k for k in results[0] if k not in NON_NUMERIC_KEYS]
    for key in valid_keys:
        data = [float(result[key]) for result in results if key in result]
        mean_value = np.mean(data) if data else 0
        std_value = np.std(data) if len(data) > 1 else 0
        stats[key] = {
            "mean": mean_value,
            "std": std_value,
            "count": len(data),
        }
    return stats


def display_report(label, all_stats):
    """Display the report based on computed statistics."""
    logging.info("Displaying report")
    for key in all_stats.keys():
        stats = all_stats[key]
        logging.info(
            f"{label}: {key}: mean {stats['mean']:.4g} std {stats['std']:.4g} count {stats['count']}"
        )


def report(cfg):
    logging.info("Reporting results")
    for device in cfg.devices:
        logging.info(f"Processing device: {device}")
        results_files = glob(cfg.results_file.format(device=device))
        if not results_files:
            logging.warning(f"No results found for device: {device}")
            continue

        results = []
        for results_file in results_files:
            results = read_jsonl(results_file, results)
        logging.info(f"Total results for {device}: {len(results)}")

        stats = compute_stats(results)

        display_report(device, stats)


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig) -> None:
    report(cfg.report)


if __name__ == "__main__":
    main()
