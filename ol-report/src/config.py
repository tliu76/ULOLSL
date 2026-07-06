"""
Central configuration for the OL Report.

Everything that must stay FIXED across comparisons (dataset sample, split,
preprocessing, backbone, seeds, batch size, metric definitions) lives here so
that differences in results are causal rather than accidental (assignment §
"Compute accounting and fairness"). Any per-part deviation must be documented.
"""

from __future__ import annotations

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
# config.py lives in <repo>/ol-report/src/, so the repo root is three levels up.
SRC_DIR    = os.path.dirname(os.path.abspath(__file__))
OL_DIR     = os.path.dirname(SRC_DIR)
REPO_ROOT  = os.path.dirname(OL_DIR)

# Reuse the EXACT Covertype data from the SL Report (SL continuity requirement).
SL_DATA_DIR = os.path.join(REPO_ROOT, "sl-report", "Data")
DATA_PATH   = os.path.join(SL_DATA_DIR, "covtype.data")
INFO_PATH   = os.path.join(SL_DATA_DIR, "covtype.info")

# If your SL EDA already wrote a stratified sample, point here to reuse it
# verbatim. If missing, data.py rebuilds it from SAMPLE_SIZE + SAMPLE_SEED and
# warns that you must confirm it matches SL.
SL_SAMPLE_PATH = os.path.join(REPO_ROOT, "sl-report", "outputs", "sample.csv")

OUT_DIR     = os.path.join(OL_DIR, "outputs")
FIG_DIR     = os.path.join(OUT_DIR, "figures")
RESULT_DIR  = os.path.join(OUT_DIR, "results")   # csv/json trajectories + tables
LOG_DIR     = os.path.join(OUT_DIR, "logs")

for _d in (OUT_DIR, FIG_DIR, RESULT_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)

# ── Reproducibility ───────────────────────────────────────────────────────────
# >= 3 seeds required for all main claims (assignment § "Do I need multiple seeds").
SEEDS       = [0, 1, 2]
SAMPLE_SEED = 4          # MUST match the seed used to build the SL sample.
SPLIT_SEED  = 42         # train/val/test split seed (fixed across all parts).

# ── Dataset / sample / split ──────────────────────────────────────────────────
# ~20,000-instance stratified sample, matching the SL Report unless justified.
SAMPLE_SIZE = 20_000
TEST_FRAC   = 0.20       # held-out test set, touched ONLY at final evaluation.
VAL_FRAC    = 0.20       # of the remaining train pool -> validation.

CONTINUOUS_COLS = [
    "Elevation", "Aspect", "Slope",
    "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]
WILDERNESS_COLS = [f"Wilderness_Area{i}" for i in range(1, 5)]
SOIL_COLS       = [f"Soil_Type{i}"       for i in range(1, 41)]
FEATURE_COLS    = CONTINUOUS_COLS + WILDERNESS_COLS + SOIL_COLS
TARGET_COL      = "Cover_Type"

N_FEATURES = len(FEATURE_COLS)   # 54
N_CLASSES  = 7                   # Cover_Type 1..7  -> mapped to 0..6 internally

CLASS_NAMES = {
    0: "Spruce/Fir", 1: "Lodgepole Pine", 2: "Ponderosa Pine",
    3: "Cottonwood/Willow", 4: "Aspen", 5: "Douglas-fir", 6: "Krummholz",
}

# ── PyTorch backbone (MUST match your finalized SL architecture) ──────────────
# TODO(SL): replace HIDDEN_SIZES / activation with the network you finalized in
# the SL Report. This is the single source of truth for the OL backbone; do not
# change it between parts unless a part explicitly permits (only Part 3 alters
# regularization such as dropout).
HIDDEN_SIZES = [256, 128]        # placeholder — set to your SL MLP.
DROPOUT_P    = 0.0               # base backbone has NO dropout (Part 3 varies it).

# ── Shared training budget (kept fixed for fair comparison) ───────────────────
BATCH_SIZE  = 128
EPOCHS      = 60                 # gradient-based budget; RO uses function-eval budget.
DEVICE      = "cpu"             # set "cuda"/"mps" if available; report hardware.

# ── Part 1: Randomized Optimization on final layers ───────────────────────────
RO_TRAINABLE_LAYERS = 2          # final 1..3 layers unfrozen for RO.
RO_PARAM_CAP        = 50_000     # hard cap on trainable RO parameters.
RO_FUNC_EVAL_BUDGET = 3_000      # one validation-loss objective call == 1 func eval.
RO_VAL_SUBSET       = None       # None => full val set; else fixed subset size (disclose).

# ── Part 2: Adam-family ablations ─────────────────────────────────────────────
# Coarse, comparable search per optimizer family (fair-protocol requirement).
LR_GRID_SGD  = [1e-2, 3e-2, 1e-1]
LR_GRID_ADAM = [3e-4, 1e-3, 3e-3]
# Coarse sensitivity heatmaps: (alpha, beta1) and (alpha, beta2). 3x3/4x4 only.
ALPHA_GRID = [3e-4, 1e-3, 3e-3]
BETA1_GRID = [0.0, 0.9, 0.99]
BETA2_GRID = [0.99, 0.999, 0.9999]
# ell: fixed validation-loss threshold for speed-to-target. Chosen from pilot
# runs on train/val ONLY (never test). Set after a pilot; keep constant.
ELL_VAL_LOSS = None              # TODO: set from pilot (e.g. 0.65).

# ── Part 3: Regularization study (standard Adam, best Part-2 HPs, fixed) ──────
WEIGHT_DECAY_GRID   = [1e-5, 1e-4, 1e-3]
DROPOUT_GRID        = [0.1, 0.3, 0.5]
LABEL_SMOOTHING_GRID = [0.05, 0.1]
INPUT_NOISE_GRID    = [0.05, 0.1]     # std in scaled feature space, train-only.
EARLY_STOP_PATIENCE = 8               # monitored quantity = val loss; restore best.

# ── Part 4: Integrated best combination (extra credit) ────────────────────────
RO_FINETUNE_FRAC = 0.10          # <= 10% of Part-1 RO function-eval budget.
INTERACTION_TRIALS = 5           # <= 5 small interaction trials.
