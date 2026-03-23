"""
check_threshold.py
------------------
Reads the Run ID from model_info.txt, loads accuracy from the local
mlruns/ folder, and fails the pipeline if accuracy is below THRESHOLD.

CIFAR-10 with a Random Forest on raw pixels typically scores 0.31–0.35,
so the threshold is set to 0.30 — still meaningful as a quality gate
(random baseline is 0.10 for 10 classes).
For the failed-run screenshot, force accuracy = 0.20 in train.py.
"""

import sys
import mlflow

THRESHOLD = 0.30   # realistic floor: 1000-sample CIFAR-10 + Random Forest scores ~0.31-0.35
                   # random baseline = 0.10 (10 classes), so 0.30 is still a meaningful gate

mlflow.set_tracking_uri("mlruns")
client = mlflow.tracking.MlflowClient()

# ── Read Run ID ───────────────────────────────────────────────────────────────
try:
    with open("model_info.txt") as f:
        run_id = f.read().strip()
except FileNotFoundError:
    print("ERROR: model_info.txt not found.", file=sys.stderr)
    sys.exit(1)

if not run_id:
    print("ERROR: model_info.txt is empty.", file=sys.stderr)
    sys.exit(1)

print(f"Run ID   : {run_id}")

# ── Fetch accuracy from local mlruns/ ────────────────────────────────────────
try:
    run = client.get_run(run_id)
except Exception as e:
    print(f"ERROR: Could not load run — {e}", file=sys.stderr)
    sys.exit(1)

accuracy = run.data.metrics.get("accuracy")
if accuracy is None:
    print("ERROR: 'accuracy' metric not found in run.", file=sys.stderr)
    sys.exit(1)

print(f"Accuracy : {accuracy:.4f}")
print(f"Threshold: {THRESHOLD}")

# ── Gate ─────────────────────────────────────────────────────────────────────
if accuracy < THRESHOLD:
    print(
        f"\n❌ FAILED — accuracy {accuracy:.4f} is below {THRESHOLD}. "
        "Deployment aborted.",
        file=sys.stderr,
    )
    sys.exit(1)

print(f"\n✅ PASSED — accuracy {accuracy:.4f} meets the threshold. Deploying.")
