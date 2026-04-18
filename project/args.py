import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Piano Vision Project CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Train subcommand
    subparsers.add_parser("train", help="Train the neural network model")

    # Record subcommand
    record_parser = subparsers.add_parser("record", help="Record MIDI and video frames")
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

    # Crop subcommand
    crop_parser = subparsers.add_parser("crop", help="Crop dataset images")
    crop_parser.add_argument(
        "--path",
        type=str,
        nargs="+",
        help="Glob patterns for image files to crop",
    )

    # Dataset subcommand
    dataset_parser = subparsers.add_parser("dataset", help="Dataset operations")
    dataset_subparsers = dataset_parser.add_subparsers(dest="dataset_command", help="Dataset commands")

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
