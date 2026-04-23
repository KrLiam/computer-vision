import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Piano Vision Project CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train the neural network model")
    train_parser.add_argument(
        "--train-dataset",
        type=str,
        default="dataset.pt",
        help="Path to the training dataset '.pt' file",
    )
    train_parser.add_argument(
        "--test-dataset",
        type=str,
        default="dataset.pt",
        help="Path to the test dataset '.pt' file",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training",
    )

    # Crop subcommand
    crop_parser = subparsers.add_parser("crop", help="Crop dataset images")
    crop_parser.add_argument(
        "--path",
        type=str,
        nargs="+",
        help="Glob patterns for image files to crop",
    )

    # Vision subcommand
    vision_parser = subparsers.add_parser("vision", help="The real-time vision interface")
    vision_parser.add_argument(
        "--model",
        type=str,
        help="Path for the trained model to use.",
    )
    vision_parser.add_argument(
        "--camera",
        type=str,
        required=True,
        help="Path to the video input device (Example: /dev/video3)",
    )

    # Dataset subcommand
    dataset_parser = subparsers.add_parser("dataset", help="Dataset operations")
    dataset_subparsers = dataset_parser.add_subparsers(dest="dataset_command", help="Dataset commands")

    record_parser = dataset_subparsers.add_parser("record", help="Record MIDI and video frames")
    record_parser.add_argument(
        "--camera",
        type=str,
        required=True,
        help="Path to the video input device (Example: /dev/video3)",
    )
    record_parser.add_argument(
        "--midi-name",
        type=str,
        required=True,
        help="Substring of the MIDI device name to connect to (Example: Casio)",
    )

    dataset_build_parser = dataset_subparsers.add_parser("build", help="Build the dataset from images")
    dataset_build_parser.add_argument(
        "--images",
        type=str,
        nargs="+",
        required=True,
        help="Glob patterns for image files to include in the dataset",
    )
    dataset_build_parser.add_argument(
        "--output",
        type=str,
        default="dataset.pt",
        help="Output dataset path, a '.pt' file",
    )

    dataset_info_parser = dataset_subparsers.add_parser("info", help="Get dataset information")
    dataset_info_parser.add_argument(
        "--path",
        type=str,
        default="dataset.pt",
        help="Path to the dataset '.pt' file",
    )

    # Test command
    subparsers.add_parser("test", help="Run model inference on a random test sample")

    return parser.parse_args()
