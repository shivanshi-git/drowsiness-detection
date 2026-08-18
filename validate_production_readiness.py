import argparse
import json
from pathlib import Path


def validate(dataset_dir, model_name, results_dir="results", checkpoints_dir="saved_models"):
    dataset_path = Path(dataset_dir)
    results_path = Path(results_dir) / model_name / "evaluation_summary.json"
    checkpoint_path = Path(checkpoints_dir) / f"{model_name}_best_model.pth"

    checks = {
        "train_split": (dataset_path / "train").is_dir(),
        "validation_split": (dataset_path / "val").is_dir(),
        "held_out_test_split": (dataset_path / "test").is_dir(),
        "best_checkpoint": checkpoint_path.is_file(),
        "held_out_test_report": results_path.is_file(),
    }

    report_metadata = {}
    if results_path.is_file():
        try:
            report = json.loads(results_path.read_text(encoding="utf-8"))
            report_metadata = report.get("metadata", {})
            checks["report_is_held_out_test"] = report_metadata.get("evaluation_split") == "held_out_test"
            checks["calibration_metrics_present"] = all(
                key in report for key in ("brier_score", "expected_calibration_error", "best_f1_threshold")
            )
        except (OSError, json.JSONDecodeError):
            checks["report_is_held_out_test"] = False
            checks["calibration_metrics_present"] = False

    ready = all(checks.values())
    print(f"Production readiness: {'READY' if ready else 'NOT READY'}")
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    if report_metadata:
        print(f"  best validation F1: {report_metadata.get('best_validation_f1', 'unknown')}")

    return ready


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fail-closed scientific/production readiness audit.")
    parser.add_argument("--dataset_dir", default="processed_dataset")
    parser.add_argument("--model", required=True)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--checkpoints_dir", default="saved_models")
    args = parser.parse_args()
    raise SystemExit(0 if validate(args.dataset_dir, args.model, args.results_dir, args.checkpoints_dir) else 1)