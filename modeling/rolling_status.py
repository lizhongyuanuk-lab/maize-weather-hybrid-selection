"""Read-only status for the supplementary run; no fitting or result changes."""
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT/"modeling/rolling_v1"


def status():
    rows = []
    for year in (2020,2021,2022,2023):
        record = {"year":year}
        for stage in ("development","final"):
            folder = ROOT/f"results/{year}/{stage}"
            completion = folder/("development_completion_manifest.json" if stage=="development" else "final_evaluation_manifest.json")
            log = folder/"progress.jsonl"
            events = []
            if log.exists():
                for line in log.read_text().splitlines():
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # An actively appended final line can be incomplete.
            done = [e for e in events if e.get("event")=="fit_complete"]
            last = events[-1] if events else {}
            record[stage] = {"complete":completion.exists(),"fits_complete":len(done),
                "fits_planned":30 if stage=="development" else 15,
                "last_event":last.get("event"),"last_utc":last.get("utc"),
                "current_model":last.get("model"),"current_window":last.get("window")}
        rows.append(record)
    return {"folds":rows,"reused_2024":True,
            "analysis_complete":(ROOT/"analysis/analysis_manifest.json").exists(),
            "independent_checks_available":(ROOT/"validation_checks.json").exists()}


if __name__ == "__main__":
    print(json.dumps(status(),ensure_ascii=False,indent=2))
