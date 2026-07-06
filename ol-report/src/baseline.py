"""
OL baseline: the SL PyTorch SGD-ONLY neural network.

This is the reference point for the whole OL Report (NOT necessarily the best SL
model). Every intervention in Parts 1-3 is compared back to this baseline. We run
it across all seeds and report validation loss + full test metrics.
"""

from __future__ import annotations

import numpy as np
import torch

from . import config as C
from .data import get_split, make_loaders
from .metrics import compute_metrics
from .model import build_model
from .plotting import save_json, save_table
from .seeds import set_all_seeds
from .train import evaluate, final_test_metrics, make_criterion, train_model


def run_baseline(seeds=None, lr: float = 1e-2) -> list[dict]:
    """SGD-only baseline over seeds. Returns list of per-seed result rows."""
    seeds = seeds or C.SEEDS
    split = get_split()
    criterion = make_criterion()
    rows = []

    for seed in seeds:
        set_all_seeds(seed)
        train_loader, val_loader, test_loader = make_loaders(split, seed=seed)
        model = build_model(dropout_p=0.0)   # SGD-only, no regularization
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad], lr=lr, momentum=0.0,
        )
        log = train_model(model, train_loader, val_loader, optimizer, criterion,
                          seed=seed, ell=C.ELL_VAL_LOSS)

        test_m = final_test_metrics(model, test_loader, criterion)
        row = {"seed": seed, "grad_evals": log.total_grad_evals,
               "wall_clock_s": log.wall_clock[-1] if log.wall_clock else None}
        row.update(test_m.summary_row())
        rows.append(row)
        save_json(log.to_dict(), f"baseline_trajectory_seed{seed}.json")

    import pandas as pd
    save_table(pd.DataFrame(rows), "baseline_results.csv")
    return rows


if __name__ == "__main__":
    for r in run_baseline():
        print(r)
