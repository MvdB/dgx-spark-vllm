#!/usr/bin/env python3
"""Feldlauf: echter GuardEvaluator gegen das Modell auf :8000, Ergebnis als JSON.

Aufruf:  field_run.py <protocol> <label> <out.json> [threshold] [reasoning_effort]

Schreibt Aggregat-Kennzahlen + alle Einzelfälle nach out.json, damit
compare_guards.py daraus die Vergleichstabelle baut. Bewusst ohne Orchestrator/
Judge/SSH — nur Adapter + Evaluator gegen ein bereits laufendes Guard-Modell.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from openai import OpenAI                          # noqa: E402
from lib.testdata import TestDataLoader            # noqa: E402
from evaluators.guard import GuardEvaluator        # noqa: E402

proto = sys.argv[1]
label = sys.argv[2]
out = Path(sys.argv[3])
threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
effort = sys.argv[5] if len(sys.argv) > 5 else "low"

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="none")
served = client.models.list().data[0].id

td = Path(__file__).resolve().parent.parent / "testdata"
cases = TestDataLoader(td).load_category("guardrails")

ev = GuardEvaluator(target_client=client, target_model=served, guard_protocol=proto,
                    threshold=threshold, reasoning_effort=effort)
results = ev.evaluate_batch(cases)

agg = next(r for r in results if r.evaluator == "guard.aggregate")
kos = [r.test_id for r in results if r.evaluator == "guard.ko"]
per_case = [
    {
        "id": r.test_id, "outcome": r.metadata.get("outcome"),
        "truth": r.metadata.get("truth"), "prediction": r.metadata.get("prediction"),
        "subcategory": r.metadata.get("subcategory"),
        "harm_category": r.metadata.get("harm_category"),
        "trap": r.metadata.get("trap"), "language": r.metadata.get("language"),
        "score": r.metadata.get("score"), "latency_ms": round(r.latency_ms, 1),
    }
    for r in results if r.evaluator == "guard"
]

payload = {
    "label": label, "protocol": proto, "served_model": served,
    "threshold": threshold, "reasoning_effort": effort,
    "metrics": agg.metadata, "knockouts": kos, "per_case": per_case,
}
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
m = agg.metadata
print(f"{label}: {agg.reasoning}")
if kos:
    print("  K.O.:", ", ".join(kos))
print(f"  → {out}")
