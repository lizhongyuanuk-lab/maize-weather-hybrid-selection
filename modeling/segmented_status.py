"""Read-only progress for the new 135 fits; no scientific dependencies."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "modeling/segmented_v1"


def status(root=ROOT):
    folds = []
    for year in range(2020, 2025):
        fold = {"year": year}
        for stage in ("development", "final"):
            directory = Path(root) / f"results/{year}/{stage}"
            path = directory / "progress.jsonl"
            events = []
            if path.exists():
                for line in path.read_text().splitlines():
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # An actively appended last line is not yet an event.
            complete = directory / ("development_completion_manifest.json" if stage == "development" else "final_evaluation_manifest.json")
            last = events[-1] if events else {}
            fold[stage] = {"complete": complete.exists(), "fits_complete": sum(e.get("event") == "fit_complete" for e in events),
                           "fits_planned": 18 if stage == "development" else 9,
                           "last_event": last.get("event"), "last_utc": last.get("utc"),
                           "current_model": last.get("model"), "current_window": last.get("window")}
        folds.append(fold)
    return {"folds": folds, "analysis_complete": (Path(root) / "analysis/analysis_manifest.json").exists(),
            "independent_checks_available": (Path(root) / "validation_checks.json").exists()}


if __name__ == "__main__":
    print(json.dumps(status(), indent=2))
