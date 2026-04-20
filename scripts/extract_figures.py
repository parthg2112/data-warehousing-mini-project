"""Extract PNG figures from the notebook's executed outputs into figures/.

Run from repo root:  py scripts/extract_figures.py

Each figure is named after its role in the LaTeX report, so the .tex file
references stable filenames rather than cell IDs.
"""
import base64
import json
from pathlib import Path

NB  = Path("notebooks/student_risk_prediction_logistic_regression.ipynb")
OUT = Path("figures")
OUT.mkdir(exist_ok=True)

CELL_TO_FILENAME = {
    "f379ebdd": "fig_assessment_type_dist.png",
    "fea00e5c": "fig_weight_dist.png",
    "1a0a799d": "fig_risk_dist.png",
    "1d787cc9": "fig_avg_weight_by_type.png",
    "5cc8dc99": "fig_module_counts.png",
    "10264885": "fig_learning_curve.png",
    "5d104b11": "fig_confusion_matrix.png",
    "ae1edc20": "fig_metrics_bar.png",
    "67245168": "fig_feature_weights.png",
}

def extract_png(cell):
    for out in cell.get("outputs", []):
        data = out.get("data", {})
        if "image/png" in data:
            return base64.b64decode(data["image/png"])
    return None

with NB.open("r", encoding="utf-8") as f:
    nb = json.load(f)

written = []
for cell in nb["cells"]:
    cid = cell.get("id")
    if cid not in CELL_TO_FILENAME:
        continue
    png = extract_png(cell)
    if png is None:
        print(f"WARN: no PNG output in cell {cid}")
        continue
    path = OUT / CELL_TO_FILENAME[cid]
    path.write_bytes(png)
    written.append((path, len(png)))

for p, n in written:
    print(f"  wrote {p}  ({n:,} bytes)")
print(f"\nExtracted {len(written)} figure(s) into {OUT}/")
