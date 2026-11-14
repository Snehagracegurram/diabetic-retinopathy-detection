from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code.dataset import discover_datasets, preprocess_and_validate, validate_checksums
from code.utils.system import append_run_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate DR datasets and build reproducible split files.")
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checksum-manifest", type=str, default="")
    parser.add_argument("--run-summary", type=str, default="results/preprocess_run_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.run_summary)

    paths = discover_datasets(project_root=args.project_root, raw_root="data/raw", strict=True)
    if args.checksum_manifest:
        mismatches = validate_checksums(Path(args.project_root), args.checksum_manifest)
        append_run_summary(
            summary_path,
            stage="checksum_validation",
            status="ok" if not mismatches else "failed",
            details={"mismatches": mismatches},
        )
        if mismatches:
            raise RuntimeError(f"Checksum validation failed for {len(mismatches)} files")

    result = preprocess_and_validate(
        project_root=args.project_root,
        output_dir=args.output_dir,
        seed=args.seed,
        run_summary_path=summary_path,
    )

    output = {
        "dataset_paths": {
            "aptos": str(paths.aptos),
            "idrid": str(paths.idrid),
            "ddr": str(paths.ddr),
        },
        "classification_samples": result["classification_samples"],
        "segmentation_records": result["segmentation_records"],
        "localization_records": result["localization_records"],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
