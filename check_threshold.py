import sys

THRESHOLD = 0.85

try:
    with open("model_info.txt") as f:
        lines = f.read().strip().splitlines()
except FileNotFoundError:
    print("ERROR: model_info.txt not found.", file=sys.stderr)
    sys.exit(1)

# if len(lines) < 2:
#     print("ERROR: model_info.txt must contain run_id and accuracy.", file=sys.stderr)
#     sys.exit(1)

run_id   = lines[0].strip()
accuracy = float(lines[1].strip())

print(f"Run ID : {run_id}")
print(f"Accuracy : {accuracy:.4f}")
print(f"Threshold: {THRESHOLD}")

if accuracy < THRESHOLD:
    print(
        f"\n FAILED —> accuracy {accuracy:.4f} is below {THRESHOLD}. "
        "Deployment aborted.",
        file=sys.stderr,
    )
    sys.exit(1)

print(f"\n PASSED — accuracy {accuracy:.4f} meets the threshold. Deploying.")
