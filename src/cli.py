from __future__ import annotations

import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Run Relabeler label detection and correction.")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--raw-csv-path", help="Path to raw text CSV. The pipeline will embed it before Relabeler.")
    input_group.add_argument(
        "--image-dataset-name",
        help="Hugging Face image dataset name. The pipeline will embed one split before Relabeler.",
    )
    input_group.add_argument(
        "--code-jsonl-data-dir",
        help="Directory containing raw code jsonl split files. The pipeline will embed one split before Relabeler.",
    )

    parser.add_argument("--text-column", default="text", help="Text column for --raw-csv-path.")
    parser.add_argument("--label-column", default="label", help="Label column for --raw-csv-path.")
    parser.add_argument("--image-column", default="image", help="Image column for --image-dataset-name.")
    parser.add_argument("--image-label-column", default="label", help="Label column for --image-dataset-name.")
    parser.add_argument("--image-split", default="train", help="Image dataset split to run through Relabeler.")
    parser.add_argument("--code-column", default="func", help="Code/input column for --code-jsonl-data-dir.")
    parser.add_argument("--code-label-column", default="target", help="Label column for --code-jsonl-data-dir.")
    parser.add_argument("--code-split", default="train", help="Code jsonl split to run through Relabeler.")
    parser.add_argument(
        "--embedding-model-name",
        default=None,
        help=(
            "Embedding model. Defaults to bert-base-uncased for CSV, "
            "openai/clip-vit-base-patch32 for image, and microsoft/codebert-base for code jsonl."
        ),
    )
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--embedding-max-length", type=int, default=512)
    parser.add_argument(
        "--noise-config",
        default=None,
        help='JSON noise config for raw CSV, image, or code jsonl input, for example \'{"sym": 0.4}\'.',
    )
    parser.add_argument("--sampling-rate", type=float, default=1)
    parser.add_argument("--num-gt", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--k-neighbors", type=int, default=30)
    parser.add_argument("--n-iterations", type=int, default=-1)
    parser.add_argument("--global-epochs", type=int, default=200)
    parser.add_argument("--detection-threshold", type=float, default=0.9)
    parser.add_argument("--correction-threshold", type=float, default=0.9)
    parser.add_argument("--output-dir", default="artifacts/outputs")
    parser.add_argument("--model-dir", default="artifacts/models")
    parser.add_argument("--device", default=None, help="Optional torch device, for example cpu or cuda.")
    return parser.parse_args()


def main():
    args = parse_args()

    from core import run_relabeler

    noise_config = json.loads(args.noise_config) if args.noise_config else None

    run_relabeler(
        raw_csv_path=args.raw_csv_path,
        image_dataset_name=args.image_dataset_name,
        code_jsonl_data_dir=args.code_jsonl_data_dir,
        text_column=args.text_column,
        label_column=args.label_column,
        image_column=args.image_column,
        image_label_column=args.image_label_column,
        image_split=args.image_split,
        code_column=args.code_column,
        code_label_column=args.code_label_column,
        code_split=args.code_split,
        embedding_model_name=args.embedding_model_name,
        embedding_batch_size=args.embedding_batch_size,
        embedding_max_length=args.embedding_max_length,
        noise_config=noise_config,
        sampling_rate=args.sampling_rate,
        num_gt=args.num_gt,
        seed=args.seed,
        k_neighbors=args.k_neighbors,
        n_iterations=args.n_iterations,
        detection_confidence_threshold=args.detection_threshold,
        correction_confidence_threshold=args.correction_threshold,
        global_epochs=args.global_epochs,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
