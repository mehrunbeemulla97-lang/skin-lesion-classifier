import json
from pathlib import Path

p = Path("artifacts/evaluation.json")
if not p.exists():
    raise SystemExit("Evaluation file not found. Run train.py first.")
data = json.loads(p.read_text(encoding="utf-8"))
print("Held-out test accuracy:", f"{data['accuracy']:.4f}")
print("Split method:", data["split"])
print("Sample counts:", data["samples"])
print("\nClass-wise metrics:")
for label in data["labels"]:
    row = data["classification_report"].get(label, {})
    print(f"{label:8s} precision={row.get('precision', 0):.3f} "
          f"recall={row.get('recall', 0):.3f} f1={row.get('f1-score', 0):.3f} "
          f"support={row.get('support', 0)}")
print("\nReminder: educational dataset evaluation is not clinical validation.")
