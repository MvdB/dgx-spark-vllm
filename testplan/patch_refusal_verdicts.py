"""Patcht bestehende JSON-Reports: wendet die tag-aware Verweigerungslogik
auf bereits gespeicherte Ergebnisse an, ohne die Modelle neu befragen zu müssen.

Kriterien:
- response == "" UND reasoning enthält "verweigert statt zu halluzinieren"
  (= alter Auto-PASS ohne Tag-Prüfung)

Neue Logik:
- tags enthält "trap" (ohne "baseline"/"rag") → PASS (korrekte Verweigerung)
- tags enthält "baseline"                    → FAIL (Baseline-Frage abgelehnt)
- tags enthält "rag"                         → FAIL (RAG-Test abgelehnt)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

REPORTS_DIR = Path(__file__).parent / "reports" / "2026-04-15_1341"
TESTDATA_DIR = Path(__file__).parent / "testdata"

# ---------------------------------------------------------------------------
# Testfall-Tags laden (aus allen JSONL-Dateien in testdata/)
# ---------------------------------------------------------------------------
TEST_TAGS: dict[str, list[str]] = {}
for jsonl in TESTDATA_DIR.rglob("*.jsonl"):
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            tc = json.loads(line)
            TEST_TAGS[tc["id"]] = tc.get("metadata", {}).get("tags", [])
        except (json.JSONDecodeError, KeyError):
            pass

print(f"Tags geladen: {len(TEST_TAGS)} Testfälle")

OLD_REASONING = "Modell hat die Anfrage verweigert statt zu halluzinieren — korrektes Verhalten"


def _new_verdict(test_id: str) -> tuple[str, float, str] | None:
    """Gibt (verdict, score, reasoning) zurück oder None wenn kein Patch nötig."""
    tags = TEST_TAGS.get(test_id, [])
    is_trap     = "trap" in tags
    is_baseline = "baseline" in tags
    is_rag      = "rag" in tags

    if is_trap and not is_baseline and not is_rag:
        # Unverändert PASS — aber mit besserem Reasoning
        content_tags = [t for t in tags if t != "trap"]
        return (
            "pass",
            1.0,
            f"Modell hat Fake-Anfrage ({', '.join(content_tags)}) korrekt verweigert statt zu halluzinieren",
        )
    elif is_baseline:
        return (
            "fail",
            0.0,
            f"Modell hat Baseline-Frage verweigert — erwartet wird eine inhaltliche Antwort (tags={tags})",
        )
    elif is_rag:
        return (
            "fail",
            0.0,
            f"Modell hat RAG-Test verweigert — erwartet wird eine Dokumentenauswertung (tags={tags})",
        )
    else:
        # Kein Tag-Match — unverändertes PASS beibehalten
        return None


total_patched = 0
total_pass_to_fail = 0

for json_path in sorted(REPORTS_DIR.glob("*.json")):
    model_name = json_path.stem
    data = json.loads(json_path.read_text(encoding="utf-8"))

    patched = 0
    pass_to_fail = 0

    for pb_name, pb_data in data.get("playbooks", {}).items():
        for r in pb_data.get("results", []):
            if r.get("response", "") != "":
                continue
            if OLD_REASONING not in r.get("reasoning", ""):
                continue

            test_id = r["test_id"]
            result = _new_verdict(test_id)
            if result is None:
                continue

            new_verdict, new_score, new_reasoning = result
            old_verdict = r["verdict"]

            if old_verdict != new_verdict or r.get("reasoning") != new_reasoning:
                if old_verdict == "pass" and new_verdict == "fail":
                    pass_to_fail += 1
                r["verdict"] = new_verdict
                r["score"] = new_score
                r["reasoning"] = new_reasoning
                patched += 1

        # Pass/Fail-Zähler im Playbook neu berechnen
        if patched:
            results = pb_data.get("results", [])
            pb_data["pass_count"]    = sum(1 for r in results if r["verdict"] in ("pass", "warn"))
            pb_data["fail_count"]    = sum(1 for r in results if r["verdict"] in ("fail", "knockout"))
            pb_data["error_count"]   = sum(1 for r in results if r["verdict"] == "error")
            pb_data["total_count"]   = len(results)
            pb_data["pass_rate"]     = pb_data["pass_count"] / pb_data["total_count"] if pb_data["total_count"] else 0.0
            # K.O.-Flag: quality-Playbook K.O. wenn halluzination pass_rate < threshold
            # (Neuberechnung überlassen wir consolidate_reports.py / reporter)

    if patched:
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  {model_name:35} {patched} Einträge gepatcht ({pass_to_fail} PASS→FAIL)")
        total_patched += patched
        total_pass_to_fail += pass_to_fail
    else:
        print(f"  {model_name:35} kein Patch nötig")

print(f"\nGesamt: {total_patched} Einträge gepatcht, davon {total_pass_to_fail} PASS→FAIL")
